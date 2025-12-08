from typing import Optional
from collections import abc
import torch
import heapq
import time
import triton
import triton.language as tl

from .common import Sequence, SequenceStatus


@triton.jit
def store_kvcache_kernel(
    k_ptr,
    k_stride,
    v_ptr,
    v_stride,
    k_cache_ptr,
    v_cache_ptr,
    kv_indices_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0)
    kv_index = tl.load(kv_indices_ptr + idx)
    if kv_index == -1:
        return
    k_offsets = idx * k_stride + tl.arange(0, BLOCK_SIZE)
    v_offsets = idx * v_stride + tl.arange(0, BLOCK_SIZE)
    k = tl.load(k_ptr + k_offsets)
    v = tl.load(v_ptr + v_offsets)
    cache_offsets = kv_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(k_cache_ptr + cache_offsets, k)
    tl.store(v_cache_ptr + cache_offsets, v)


def store_kvcache(k: torch.Tensor, v: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, kv_indices: torch.Tensor):
    num_tokens, num_heads, head_dim = k.shape
    BLOCK_SIZE = num_heads * head_dim
    assert k.stride(-1) == 1 and v.stride(-1) == 1
    assert k.stride(-2) == head_dim and v.stride(-2) == head_dim
    assert k_cache.stride(-3) == BLOCK_SIZE and v_cache.stride(-3) == BLOCK_SIZE
    assert kv_indices.numel() == num_tokens
    grid = (num_tokens,)
    store_kvcache_kernel[grid](k, k.stride(0), v, v.stride(0), k_cache, v_cache, kv_indices, BLOCK_SIZE) # type: ignore

# KVCachePool should be instantiated in each worker
class KVCachePool:
    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
        num_tokens: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        self.dtype = dtype
        self.device = device
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or num_layers - 1
        self.create_cache_pool()

    def create_cache_pool(self):
        self.k_cache = torch.empty(
            (self.num_layers, self.num_tokens, self.num_heads, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.v_cache = torch.empty(
            (self.num_layers, self.num_tokens, self.num_heads, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )

    def get_kv_cache(self, layer: int, index: Optional[torch.Tensor] = None):
        if index is None:
            k_cache = self.k_cache[layer - self.start_layer]
            v_cache = self.v_cache[layer - self.start_layer]
        else:
            k_cache = self.k_cache[layer - self.start_layer, index]
            v_cache = self.v_cache[layer - self.start_layer, index]
        return k_cache, v_cache
    
    def set_kv_cache(self, layer: int, index: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor):
        # self.k_cache[layer - self.start_layer, index] = k_cache
        # self.v_cache[layer - self.start_layer, index] = v_cache
        store_kvcache(
            k_cache,
            v_cache,
            self.k_cache[layer - self.start_layer],
            self.v_cache[layer - self.start_layer],
            index
        )

class KVCacheAllocator:
    def __init__(
        self,
        size: int,
    ):
        self.size = size
        # Use a stack (LIFO) for O(1) amortized alloc/free.
        self.free_slots = list(reversed(range(size)))

    def alloc(self, need_size: int):
        if need_size > len(self.free_slots):
            return None
        select_index = [self.free_slots.pop() for _ in range(need_size)]
        return select_index

    def free(self, free_index: abc.Iterable[int]):
        self.free_slots.extend(free_index)

class RadixTreeNode:
    def __init__(
        self,
        # child dict 的 key 等于每个子节点 key 的开头 token
        children: Optional[dict[int, "RadixTreeNode"]] = None,
        parent: Optional["RadixTreeNode"] = None,
        key: tuple[int, ...] = tuple(),
        value: tuple[int, ...] = tuple(),
        ref_count: int = 0
    ):
        # 避免可变默认参数导致多个节点共享 children 字典
        self.children: dict[int, RadixTreeNode] = children if children is not None else {}
        self.parent = parent
        self.key = key
        self.value = value

        self.access_time = time.monotonic()
        self.ref_count = ref_count

    def __lt__(self, other: "RadixTreeNode"):
        return self.access_time < other.access_time


def get_prefix_len(key1: abc.Sequence[int], key2: abc.Sequence[int]):
    prefix_len = 0
    while prefix_len < min(len(key1), len(key2)):
        if key1[prefix_len] != key2[prefix_len]:
            break
        prefix_len += 1
    return prefix_len


class RadixTree:
    def __init__(
        self,
        kv_cache_allocator: KVCacheAllocator
    ):
        self.root = RadixTreeNode(ref_count=1)
        self.kv_cache_allocator = kv_cache_allocator

    def match_prefix(self, key: list[int]):
        node = self.root
        child_key = key[0]
        value: list[int] = []

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.access_time = time.monotonic()

            # get max matched prefix length
            prefix_len = get_prefix_len(child.key, key)
            
            if prefix_len < len(child.key):
                new_node = self._split_node(child, prefix_len)
                value.extend(new_node.value)
                node = new_node
                break
            else:
                value.extend(child.value)
                node = child
                key = key[prefix_len:]
                child_key = key[0] if len(key) > 0 else -1

        cache_indices = value
        last_prefix_node = node
        return cache_indices, last_prefix_node
    
    def insert(self, key: list[int], value: list[int]):
        node = self.root
        child_key = key[0]
        total_prefix_len = 0

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.access_time = time.monotonic()

            # get max matched prefix length
            prefix_len = get_prefix_len(child.key, key)
            total_prefix_len += prefix_len

            key = key[prefix_len:]
            value = value[prefix_len:]
            child_key = key[0] if len(key) > 0 else -1

            if prefix_len < len(child.key):
                new_node = self._split_node(child, prefix_len)
                node = new_node
                break
            else:
                node = child
        # last prefixed node
        last_prefix_node = node
        last_node = node

        # if we have unmatched keys, create a new node
        if len(key) > 0:
            last_node = self._add_node(
                parent=last_prefix_node,
                key=tuple(key),
                value=tuple(value)
            )
        return total_prefix_len, last_node
    
    def inc_ref(self, node: RadixTreeNode):
        while node != self.root:
            node.ref_count += 1
            node = node.parent or self.root

    def dec_ref(self, node: RadixTreeNode):
        while node != self.root:
            node.ref_count -= 1
            node = node.parent or self.root

    # TODO: need optimization, maybe we can introduce a Evictor containing a link list
    def evict(self, num_tokens: int):
        leaves = self._get_leaf_nodes()
        heapq.heapify(leaves)

        num_evicted = 0
        # len(leaves) == 0: no evictable node
        while num_evicted < num_tokens and len(leaves) > 0:
            evict_node = heapq.heappop(leaves)
            if evict_node == self.root:
                # there is no left node can be evicted
                break
            # only evict ref_count == 0 node, meaning no request is using this node
            if evict_node.ref_count > 0:
                # there is still other request using this node
                continue

            self.kv_cache_allocator.free(evict_node.value)
            num_evicted += len(evict_node.value)

            # delete evict_node
            self._remove_node(evict_node)

            if evict_node.parent and len(evict_node.parent.children) == 0:
                heapq.heappush(leaves, evict_node.parent)

    def _get_leaf_nodes(self):
        leaf_nodes: list[RadixTreeNode] = []
        def traverse_tree(node: RadixTreeNode):
            if len(node.children) == 0:
                leaf_nodes.append(node)
                return
            for child in node.children.values():
                traverse_tree(child)
            
        traverse_tree(self.root)
        return leaf_nodes
    
    def _remove_node(self, node: RadixTreeNode):
        parent_node = node.parent
        if parent_node:
            child_key = node.key[0]
            del parent_node.children[child_key]
    
    def _add_node(
        self,
        parent: RadixTreeNode,
        key: tuple[int, ...] = tuple(),
        value: tuple[int, ...] = tuple(),
        ref_count: int = 0
    ):
        new_node = RadixTreeNode(
            parent=parent,
            key=key,
            value=value,
            ref_count=ref_count
        )
        new_node.parent = parent
        parent.children[new_node.key[0]] = new_node
        return new_node

    def _split_node(self, node: RadixTreeNode, split_len: int):
        # parent -> node ==> parent -> new_node -> node, return new_node
        new_node = RadixTreeNode(
            parent=node.parent,
            children={node.key[split_len]: node},
            key=node.key[:split_len],
            value=node.value[:split_len],
            ref_count=node.ref_count
        )
        node.parent = new_node
        node.key = node.key[split_len:]
        node.value = node.value[split_len:]

        if new_node.parent:
            new_child_key = new_node.key[0]
            new_node.parent.children[new_child_key] = new_node

        return new_node

class KVCacheManager:
    def __init__(self, size: int):
        self.kv_cache_allocator = KVCacheAllocator(size)
        self.radix_tree = RadixTree(self.kv_cache_allocator)
        # seq_id -> (prefix_len, last_node)
        self.unfinished_sequences: dict[str, tuple[int, RadixTreeNode]] = {}

    def alloc_slots(self, num_slots: int):
        indices = self.kv_cache_allocator.alloc(num_slots)
        # if no space, evict some slots from radix tree
        if indices is None:
            self.radix_tree.evict(num_slots)
            indices = self.kv_cache_allocator.alloc(num_slots)
        # if still no space, raise error
        if indices is None:
            raise RuntimeError(f"Failed to allocate {num_slots} slots, KV cache is full!")
        return indices

    def free_slots(self, indices: abc.Iterable[int]):
        self.kv_cache_allocator.free(indices)

    def cache_sequence(self, seq: Sequence):
        token_ids = seq.token_ids
        kv_indices = seq.kv_indices

        new_prefix_len, new_last_node = self.radix_tree.insert(token_ids, kv_indices)
        old_prefix_len, old_last_node = self.unfinished_sequences.get(
            seq.seq_id, (0, self.radix_tree.root)
        )

        # 相当于把 token_ids 对应的 kv_indices 取出来
        new_indices, _ = self.radix_tree.match_prefix(token_ids)

        # free the unmatched value slots
        # 这种情况发生在存在两个seq都没有插入radix tree，但是包含相同的prefix
        # seq1: [1,2,3,4], kv1: [10,11,12,13]
        # seq2: [1,2,3,5], kv2: [20,21,22,23]
        # 插入seq1后，radix tree中有[1,2,3,4] -> [10,11,12,13]
        # 插入seq2时，匹配到[1,2,3]，4和5不匹配，但是[1,2,3]对应的kv是[10,11,12]和[20,21,22]不匹配
        # 这时需要释放掉kv2中的[20,21,22]，因为这部分kv重复了，只需要保留kv1中的[10,11,12]
        # 于是最后radix tree中有[1,2,3,4] -> [10,11,12,13]和[1,2,3,5] -> [10,11,12,23]
        if new_prefix_len > old_prefix_len and old_prefix_len != 0:
            # 释放重复的kv cache slots
            self.kv_cache_allocator.free(kv_indices[old_prefix_len:new_prefix_len])
            # 更新sequence的kv_indices的重复部分
            seq.kv_indices[old_prefix_len:new_prefix_len] = new_indices[old_prefix_len:new_prefix_len]

        self.unfinished_sequences[seq.seq_id] = (len(new_indices), new_last_node)
        self.radix_tree.dec_ref(old_last_node)
        self.radix_tree.inc_ref(new_last_node)

        if seq.status == SequenceStatus.FINISHED:
            self.radix_tree.dec_ref(new_last_node)
            if seq.seq_id in self.unfinished_sequences:
                del self.unfinished_sequences[seq.seq_id]
            # kv indices 交给 radix tree 管理了，此后 seq 中的 kv_indices 可能不再有效
            seq.kv_indices.clear()
            seq.cached_kv_len = 0
