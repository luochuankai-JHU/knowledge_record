.. knowledge_record documentation master file, created by
   sphinx-quickstart on Tue July 4 21:15:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

******************
LLM
******************

学习路线
=====================
https://chatgpt.com/s/t_69accb5b2c448191a05c43ba02e2095b

可以。下面我给你一份**更简单、能直接执行的 8 周计划**。我按你的真实情况来设计：

* 你有 ML/算法背景，不是纯小白
* 你**听过一轮课，但没有形成体系**
* 你现在最需要的是：**快速补概念 + 立刻做项目 + 能转岗用**
* 你的目标不是研究大模型训练，而是**尽快具备 GenAI / LLM / Agent 岗位的可证明能力**

我先给结论：

**你这 8 周不要再广撒网看很多视频。**
主线只保留 5 类资源：

1. 一份 LLM 基础材料
2. 一份 Agent 官方材料
3. 一份 RAG 官方材料
4. 一份 MCP 官方材料
5. 两个你自己做的项目

这样最省时间，也最容易坚持。OpenAI 现在把“building agents”单独做成学习轨道；Hugging Face 也已经把学习入口拆成 LLM Course、Agents Course、MCP Course；LlamaIndex 则把 RAG 作为应用基础来组织文档。说明现在最稳的学习主线，就是这几块。([OpenAI Developers][1])

---

# 你的总路线

## 第 1 阶段：重建地图（第 1-2 周）

目标：把这些词真正搞明白
**LLM、token、embedding、prompt、structured output、tool calling、RAG、agent、workflow、guardrails、MCP**

## 第 2 阶段：做最小可用项目（第 3-4 周）

目标：从“听懂”变成“自己能写”

## 第 3 阶段：做两个能写进简历的项目（第 5-8 周）

目标：能找工作、能讲项目、能面试

---

# 你每周的时间安排

如果你在上班，我建议每周 **10 小时左右**：

* **3 小时学习概念**
* **5 小时 coding / 做项目**
* **2 小时写笔记 + 复盘**

**项目时间一定要比看课时间多。**

---

# 8 周详细执行计划

---

## Week 1：把大模型地图搭起来

### 本周目标

你要能用自己的话讲清楚下面 8 个词：

* LLM
* token
* embedding
* context window
* prompt
* structured output
* tool calling
* hallucination

### 学习内容

主线只看这 3 个：

**1. Hugging Face LLM Course**
先看 Chapter 1 的入门部分，用来把 LLM 基础概念补起来。([Hugging Face][2])

**2. Karpathy 的 LLM 总览视频**
适合把你以前模糊听过的东西重新串起来。([OpenAI][3])

**3. 你之前看过的李宏毅课程**
只回看你最模糊的部分，不要从头完整重刷。这个是你的复习材料，不是主线。

### 本周产出

你必须写一份文档，名字就叫：

**《LLM 基础术语表》**

每个词写 3 件事：

* 它是什么
* 它解决什么问题
* 它和相邻概念的区别

### 本周最小练习

写一个最简单的 Python 小脚本，做两件事：

* 输入一段文本，让模型总结
* 输入一段文本，让模型输出固定 JSON

这样你会真正感受到：

* 普通输出和 structured output 的区别
* prompt 写法会影响输出稳定性

### 本周资源

* OpenAI building agents 学习轨道（先只看概览） ([OpenAI Developers][1])
* Hugging Face LLM Course ([Hugging Face][2])
* Hugging Face Learn 总入口 ([Hugging Face][4])

---

## Week 2：把 RAG、Agent、MCP 先分清

### 本周目标

你要能说清楚：

* RAG 是什么
* agent 是什么
* workflow 和 agent 的区别
* MCP 是什么
* OpenClaw 属于哪一层

### 学习内容

这周只看 4 个资源：

**1. LlamaIndex 的 RAG 入门**
先建立正确的 RAG 心智模型。([LlamaIndex OSS Documentation][5])

**2. OpenAI Agents guide**
先看“agent 是怎么构成的”。([OpenAI Developers][6])

**3. Hugging Face Agents Course Unit 0/1**
建立 agent 的基础认识。([Hugging Face][7])

**4. Hugging Face MCP Course Unit 0/1**
建立 MCP 的正确理解。([Hugging Face][8])

MCP 官方和 Anthropic 的介绍都把它定义成：**把 AI 应用和外部工具/数据源连接起来的开放协议**。OpenAI 也已经提供了 MCP 相关的官方指南和 Apps SDK 文档，说明它已经不是“概念热词”，而是实际开发接口的一部分。([Anthropic][9])

### 本周产出

再写一份文档：

**《RAG / Agent / MCP 一页图》**

只写 4 段：

* RAG 做什么
* agent 做什么
* MCP 做什么
* OpenClaw 为什么只是一个产品/平台案例，不是基础概念

### 本周最小练习

做两个非常小的 demo：

1. 一个最小 RAG：拿 5~10 篇文档做问答
2. 一个最小 tool calling：让模型调用一个你自己写的函数

### 本周资源

* LlamaIndex RAG 入门 ([LlamaIndex OSS Documentation][5])
* OpenAI Agents guide ([OpenAI Developers][6])
* Hugging Face Agents Course ([Hugging Face][7])
* Hugging Face MCP Course ([Hugging Face][8])

---

## Week 3：做第一个真正属于你的小项目

### 本周目标

做一个**结构化信息抽取项目**。
这个项目不花哨，但非常锻炼基础。

### 项目题目

**Text → Structured JSON Extractor**

比如输入：

* 产品介绍
* 会议纪要
* 职位描述
* 新闻摘要

输出：

* 固定字段 JSON
* 缺失值标记
* 简单校验

### 你要学的内容

* prompt 结构
* structured outputs
* schema 设计
* 错误样例
* 基础测试集

OpenAI 的 agent 与 guide 文档都强调，agent/app 不是单纯聊天，而是围绕 tools、instructions、structured outputs 和 workflow 去组织。([OpenAI Developers][6])

### 本周完成标准

你至少要有：

* 20 条输入样本
* 1 个固定 schema
* 失败样例记录
* README

### 本周资源

* OpenAI Agents SDK 文档（只看基础概念） ([OpenAI Developers][10])
* OpenAI guides 入口 ([OpenAI Developers][11])

---

## Week 4：把 RAG 做到“能解释”

### 本周目标

做一个**稍微认真一点的 RAG 项目**。

### 项目题目

**LLM 学习资料问答助手**

数据源建议：

* 你这 8 周里看的课程和官方文档
* 你自己整理的 GenAI 学习资料
* 一些高质量技术文档

### 你要实现的功能

至少包括：

* 文档切 chunk
* embeddings
* retrieval top-k
* 引用来源
* 简单 answer generation
* FAQ 测试集

### 本周重点不是“跑通”

而是你要能回答：

* 为什么 chunk 这样切
* 为什么 top-k 设成这样
* 为什么答案会错
* 什么情况下会 hallucinate

LlamaIndex 的 RAG 文档本身就是围绕这些阶段组织的。([LlamaIndex OSS Documentation][5])

### 本周完成标准

* 能回答至少 20 个问题
* 每个答案都带来源
* 记录 5 个失败案例

---

## Week 5：开始真正学 Agent

### 本周目标

搞懂 agent 不等于“让模型自己无限思考”。

### 学习内容

这周只看 OpenAI agent 相关主线：

* Building agents track ([OpenAI Developers][1])
* Agents guide ([OpenAI Developers][6])
* Agent Builder / workflow 说明 ([OpenAI Developers][12])
* 实践指南 PDF / 文章 ([OpenAI][13])

这些官方材料反复强调：
**好用的 agent 是 workflow、tools、orchestration、guardrails 的组合，不是“越自动越高级”。**([OpenAI][13])

### 本周最小练习

做一个小 agent：

* 能判断用户问题是否需要查资料
* 能在需要时调用你的 RAG 工具
* 最后输出结构化回答

### 本周产出

写一篇短文：

**《workflow 和 agent 的区别》**

你以后面试很可能会被问。

---

## Week 6：做你的第一个可讲 Agent 项目

### 项目题目

**Research Assistant Agent**

输入一个问题，比如：

* 比较两个框架
* 总结一个主题
* 生成调研报告

输出：

* 先检索
* 再归纳
* 最后输出结构化报告

### 你要实现的能力

* 意图识别
* 是否调用工具
* 调用你的 RAG 模块
* 统一输出格式
* 简单日志记录

### 本周完成标准

* 至少 10 个真实问题
* 3 类错误案例
* 1 份架构说明图

这个项目会比“单纯聊天 demo”更像招聘里会认可的应用工程项目。OpenAI 的 Agents SDK 明确把 handoff、tools、trace 这些能力放在核心位置。([OpenAI Developers][10])

---

## Week 7：补 MCP，但只学到“够用”

### 本周目标

理解 MCP 的真实作用，不神化它。

### 学习内容

* Hugging Face MCP Course Unit 0/1 ([Hugging Face][8])
* Anthropic 的 MCP 介绍 ([Anthropic][9])
* OpenAI 的 MCP / connectors 指南 ([OpenAI Developers][14])
* OpenAI Apps SDK 的 MCP server 概念页 ([OpenAI Developers][15])

这些资料都指向同一个核心：
**MCP 是标准化的工具/资源接入层。**
它解决的是“每接一个系统都手写一套集成”的问题。([Anthropic][9])

### 本周最小练习

不用一上来自己写完整 MCP server。
你只要做到：

* 看懂 client / server / tool / resource 关系
* 跑通一个最小 MCP 示例
* 写一页总结：MCP 和普通 function calling 的区别

### 对 OpenClaw 的安排

这周可以花 **1~2 小时** 了解一下 OpenClaw 是怎么把 agent 产品化的。
但它不进入主线。原因是它更像具体平台，而且生态迭代快、风险也更高；你现在学它的收益，远不如学底层通用能力。([Anthropic][9])

---

## Week 8：整理作品集 + 转岗准备

### 本周目标

把前面两项目变成“可投简历的成果”。

### 你要交付的东西

你至少要整理出这 4 样：

**1. 项目 A：Structured Extractor**

* GitHub repo
* README
* 示例输入输出
* 错误案例

**2. 项目 B：Research / RAG Agent**

* GitHub repo
* 架构图
* 示例问题
* 失败案例
* 改进方向

**3. 一份 2 页学习笔记**
标题建议：

* 《我对 LLM / RAG / Agent / MCP 的理解》

**4. 一份简历项目描述**
每个项目写：

* 解决什么问题
* 技术栈
* 关键设计
* 评测方法
* tradeoff

---

# 你真正要学的内容清单

下面这份是你**必须会**的，不多，但很关键。

## A. 基础概念层

* token
* embedding
* context window
* prompt basics
* structured output
* hallucination
* temperature / deterministic output

## B. LLM 应用层

* API 调用
* prompt design
* structured outputs
* tool calling
* simple evaluation

## C. RAG 层

* chunking
* embeddings
* top-k retrieval
* rerank 的概念
* citation
* failure analysis

## D. Agent 层

* workflow
* routing
* tool use
* handoff
* trace/logging
* guardrails

## E. MCP 层

* client / server
* tools / resources
* remote MCP
* 与普通 function calling 的关系

---

# 最少资源版清单

你如果只想保留最少最有效的资源，就保留下面这些。

## 1. LLM 基础

Hugging Face LLM Course ([Hugging Face][2])

## 2. Agent 主线

OpenAI Building Agents Track ([OpenAI Developers][1])
OpenAI Agents Guide ([OpenAI Developers][6])
OpenAI Practical Guide to Building Agents ([OpenAI][13])

## 3. RAG 主线

LlamaIndex RAG Introduction ([LlamaIndex OSS Documentation][5])

## 4. Agent 课程补充

Hugging Face Agents Course ([Hugging Face][16])

## 5. MCP 主线

Hugging Face MCP Course ([Hugging Face][8])
Anthropic MCP Introduction ([Anthropic][9])
OpenAI MCP / Connectors Guide ([OpenAI Developers][14])

---

# 你每天怎么学，最简单

我建议你每次学习都用这个固定模板，别想太多。

## 每次 90 分钟

**前 20 分钟：看一个小节**
只看一个概念，不连续刷很多。

**中间 50 分钟：立刻写代码**
哪怕只是 30 行脚本。

**最后 20 分钟：写笔记**
只回答 3 个问题：

* 今天学了什么
* 我还不懂什么
* 我能不能用自己的话解释

这个方法比“先囤一堆课程再说”有效得多。

---

# 你现在最该避免的 4 件事

1. **不要再囤很多视频**
   你现在不是缺信息，是缺落实。

2. **不要把 OpenClaw 当主线**
   它适合了解，不适合优先深学。

3. **不要一开始就学复杂 finetuning**
   你当前目标是转岗和做项目，不是训基础模型。

4. **不要只看，不写**
   哪怕每天只写 30 行代码，也比只看视频强。

---

# 我给你的最终建议

如果你想要**最简单高效直接执行**，就这么做：

**前 2 周：只补概念，不贪多**
**第 3-4 周：做一个结构化抽取小项目**
**第 5-6 周：做一个 RAG / agent 项目**
**第 7 周：补 MCP 到够用**
**第 8 周：整理成简历项目**

这条路线已经够你从“半懂不懂”走到“能做项目、能讲项目”。

下一条我可以继续直接帮你做一份 **“Week 1 到 Week 2 的逐日任务清单”**，具体到每天看哪一页、写什么代码、产出什么文件。

[1]: https://developers.openai.com/tracks/building-agents/?utm_source=chatgpt.com "Building agents"
[2]: https://huggingface.co/learn/llm-course/chapter1/1?utm_source=chatgpt.com "Introduction - Hugging Face LLM Course"
[3]: https://openai.com/business/learn/?utm_source=chatgpt.com "OpenAI Learning Hub: AI Guides, Tutorials & Resources"
[4]: https://huggingface.co/learn?utm_source=chatgpt.com "Learn"
[5]: https://developers.llamaindex.ai/python/framework/understanding/rag/?utm_source=chatgpt.com "Introduction to RAG | LlamaIndex OSS Documentation"
[6]: https://developers.openai.com/api/docs/guides/agents/?utm_source=chatgpt.com "Agents | OpenAI API"
[7]: https://huggingface.co/learn/agents-course/en/unit0/introduction?utm_source=chatgpt.com "Welcome to the 🤗 AI Agents Course"
[8]: https://huggingface.co/learn/mcp-course/en/unit0/introduction?utm_source=chatgpt.com "Welcome to the 🤗 Model Context Protocol (MCP) Course"
[9]: https://www.anthropic.com/news/model-context-protocol?utm_source=chatgpt.com "Introducing the Model Context Protocol"
[10]: https://developers.openai.com/api/docs/guides/agents-sdk/?utm_source=chatgpt.com "Agents SDK | OpenAI API"
[11]: https://developers.openai.com/resources/guides/?utm_source=chatgpt.com "Guides"
[12]: https://developers.openai.com/api/docs/guides/agent-builder/?utm_source=chatgpt.com "Agent Builder | OpenAI API"
[13]: https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/?utm_source=chatgpt.com "A practical guide to building agents"
[14]: https://developers.openai.com/api/docs/guides/tools-connectors-mcp/?utm_source=chatgpt.com "MCP and Connectors | OpenAI API"
[15]: https://developers.openai.com/apps-sdk/concepts/mcp-server/?utm_source=chatgpt.com "MCP"
[16]: https://huggingface.co/agents-course?utm_source=chatgpt.com "Hugging Face Agents Course"
