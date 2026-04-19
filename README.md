# Bookcaster

Bookcaster 是一个自动化的小说播客生成工具，能够将文本小说转换为带有不同角色声音的音频播客。

## 功能特性

- **智能对话提取**：自动识别小说文本中的对话部分，并区分说话角色和旁白
- **角色特征分析**：基于AI模型分析角色的年龄、性别等特征，为不同角色定制声音
- **多角色语音合成**：利用先进的TTS技术为不同角色生成匹配的声音
- **情感语气控制**：根据不同情境调整语音的情感表达

## 工作原理

Bookcaster 通过以下步骤将小说文本转换为音频：

1. **文本解析**：从输入的文本文件中提取对话和叙述内容
2. **角色识别**：使用大语言模型识别文本中的不同角色
3. **特征分析**：分析每个角色的年龄、性别等特征
4. **脚本生成**：将原始文本转换为适合语音合成的脚本格式
5. **语音合成**：使用TTS服务为不同角色和旁白生成相应的声音
6. **音频合并**：将所有音频片段按顺序合并为完整的播客

## 依赖环境

- Python >= 3.12
- 一个兼容OpenAI API的大语言模型服务（如LMStudio）
- 一个TTS服务（目前仅支持vllm部署的Qwen3-TTS openai compatible api）

## 安装方法

```bash
# 克隆项目
git clone https://github.com/SnowFox4004/bookcaster
cd bookcaster

# 安装依赖
uv sync
```

## 使用方法

### 1. 准备输入数据
将你的小说章节文本文件放入指定目录下，确保小说文件扩展名为 `.txt`。

### 2. 配置API参数
在 `main.py` 中修改API密钥和端点地址以匹配你的本地服务：

```python
# 配置LLM服务
LLMProvider(
    model="qwen3.5-4B",      
    api_key="lmstudio",      
    base_url="http://127.0.0.1:1234/",  
)

# 配置TTS服务
Qwen3TTS(
    api_key="ttskey",  
    api_base="http://127.0.0.1:8000/v1", 
)
```
同时配置小说存放的目录

### 3. 运行Bookcaster

```bash
uv run src/bookcaster/main.py

## 输出结果

- `chapters.json` - 生成的对话脚本
- `character_traits.json` - 分析出的角色特征
- `*.mp3` - 合成的音频文件，每个章节一个文件

## 项目结构

```bash
src/
├── bookcaster/              # 主程序包
│   ├── __init__.py
│   ├── llms.py              # 大语言模型接口
│   ├── main.py              # 程序入口
│   ├── prompts.py           # 提示词模板
│   ├── trait_guesser.py     # 角色特征分析
│   ├── utils.py             # 工具函数
│   └── tts/                 # 语音合成模块
│       ├── qwen3tts.py      # Qwen3 TTS实现
│       ├── tts_utils.py     # TTS工具函数
│       └── voxcpm2.py       # VoxCPM2模型接口 (TODO)
├── README.md
└── pyproject.toml
```

## 自定义配置

你可以通过替换以下模块来自定义Bookcaster的行为：

- **LLM Provider**: 更换为不同的大语言模型
- **TTS Provider**: 更换为其他TTS引擎
- **Trait Guesser**: 实现自己的角色特征分析算法
- **Prompts**: 修改提示词以改变AI的行为

## 贡献

欢迎提交Issue和Pull Request来改进Bookcaster！
