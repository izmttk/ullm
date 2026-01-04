# Profiling

ullm 现在支持使用 torch profiler 进行性能分析。并导出 trace 数据。

## 使用方法

在启动服务时，添加 `--profile` 参数，以启用 torch profiler 功能的支持：

```bash
python -m ullm.entrypoints.api_server --model <model_name> --profile
```

## Trace 捕获
启动服务后，profiler 还没有开始捕获 trace。要开始捕获 trace，需要向服务的 `/profile` 端点发送 POST 请求：

```bash
curl -X POST http://localhost:8000/profile?action=start
# <your custom requests here>
curl -X POST http://localhost:8000/profile?action=stop
```

其中 `action` 参数可以是 `start` 或 `stop`，分别用于开始和停止 trace 捕获。当 `action=stop` 时，profiler 会将捕获的 trace 数据导出到指定位置，终端 log 中会显示必要的信息。

导出的 trace 数据包括：
1. 一个 `ullm_engine_<pid>.<timestamp>.pt.trace.json.gz`，表示 `Engine` 实例所在进程的 trace 数据。
2. 一个或多个 `ullm_worker_<rank>_TP_<tp_rank>_PP_<pp_rank>_<pid>.<timestamp>.pt.trace.json.gz`，表示各个 worker 进程的 trace 数据。

## 指定导出位置

默认情况下，trace 数据会导出到当前工作目录下的 `profiles` 文件夹中。你也可以通过设置启动参数 `--profile-dir` 来指定导出位置：

```bash
python -m ullm.entrypoints.api_server --model <model_name> --profile --profile-dir /path/to/your/profile/dir
```

## 查看 Trace 数据

建议使用 [Perfetto](https://ui.perfetto.dev/) 来查看和分析导出的 trace 数据。