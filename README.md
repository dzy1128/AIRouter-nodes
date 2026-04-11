# AIRouter Nodes

适用于 ComfyUI 的 AIRouter 图片节点集合，当前包含：

- `AIRouter Seedream Image`
- `AIRouter Gemini Image`

## 功能

- 支持文生图
- 支持最多 5 张输入图做图生图
- 支持自定义模型名
- 从环境变量读取 API Key
- 自动解析 `url`、`base64`、`bytes` 三类图片返回

## 环境变量

优先读取：

- `AIROUTER-API-KEY`

同时兼容：

- `AIROUTER_API_KEY`

## 默认接口

- `base_url`: `https://api-ai.gk.cn`
- `endpoint`: `/v1/images/generations`

## 说明

由于参考文档接口需要登录 token 才能直接读取，这个实现按用户提供的示例脚本完成，并把可能变动的字段保留为节点可配置项：

- `model`
- `response_format`
- `base_url`

如果你后续拿到文档里的精确字段枚举，可以继续把模型列表和参数选项收紧成下拉框。
