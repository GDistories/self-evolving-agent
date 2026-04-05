# 自进化 Agent 设计文档

日期：2026-04-06

## 目标

构建一个在本地运行的自进化 Agent，通过反复评测和修订，持续优化基于 Prompt 的 Prompt Attack Detection 能力。

本地程序是唯一的控制器。本地不部署大模型，而是依赖两个远端能力端点：

- `server1`：通过 `/v1/chat/completions` 提供远端大模型能力，作为 Agent 的“大脑”
- `server2`：使用 8B 模型执行评测，并返回结构化结果

第一版实现必须支持：

- 对 `system_prompt` 进行迭代优化
- 对用户侧包装模板 `user_template` 进行迭代优化
- 在运行时动态输入评测指标
- `server2` 提供异步评测任务
- `server2` 提供结果缓存
- 在本地执行“单 best candidate”的优化闭环

第一版实现不依赖：

- 写死在代码中的指标阈值
- 多分支搜索树
- `server2` 上的复杂策略学习
- 本地大模型推理

## 范围

本设计覆盖一个完整的端到端优化闭环：

1. 从一版基线 Prompt 配置开始
2. 请求 `server1` 生成新的候选版本
3. 将候选版本提交给 `server2`
4. 使用 `tp.jsonl` 和 `tn.jsonl` 执行评测
5. 根据运行时输入的动态指标进行比较
6. 决定晋级、淘汰或继续迭代

初始数据集格式固定为两个 JSONL 文件：

- `tp.jsonl`
- `tn.jsonl`

每条记录格式为：

```json
{"id": "...", "text": "..."}
```

标签由文件身份决定，而不是由每条样本显式存储。

## 架构

系统包含三个运行时角色。

### 1. 本地 Agent Controller

本地控制器负责编排完整闭环。职责包括：

- 加载基线候选版本
- 加载本轮运行所需的动态评测要求
- 调用 `server1` 生成新的候选版本
- 向 `server2` 提交评测任务
- 轮询任务状态
- 将评测结果与当前最佳版本比较
- 决定停止、继续或晋级
- 在本地保存实验历史

该组件拥有所有优化策略，以及所有本地实验记录。

### 2. Server1：候选版本生成器

`server1` 是远端大模型端点，充当 Agent 的大脑。它只负责生成候选版本更新，不保存闭环状态，也不直接判断候选质量。

职责包括：

- 接收本地控制器提供的结构化上下文
- 返回修订后的候选版本
- 以机器可解析或易解析的形式解释修改意图

### 3. Server2：评测器

`server2` 是基于 8B 模型的远端评测服务。它不负责优化策略，只负责执行评测任务、缓存结果，并返回结构化输出。

职责包括：

- 接收候选版本评测任务
- 异步执行评测
- 对等价输入的重复评测命中缓存
- 返回合并后的指标和失败样本

## Candidate 模型

Candidate 是优化过程中的原子单位，表示一版待测配置。

建议字段：

- `candidate_id`
- `parent_candidate_id`
- `system_prompt`
- `user_template`
- `mutation_note`
- `created_at`
- `created_by`
- `status`

`status` 至少应支持：

- `draft`
- `testing`
- `passed`
- `rejected`
- `best`

第一版只需要支持修改：

- `system_prompt`
- `user_template`

但接口设计应预留未来扩展能力，用于支持生成参数，例如 `temperature`、`max_tokens`、约束输出等。

## Evaluation Spec

评测标准必须在运行时输入，不能写死在代码中。

建议字段：

- `tp_path`
- `tn_path`
- `metric_config`
- `max_rounds`
- `top_k_failures_to_feedback`
- `stop_conditions`

之所以将 `metric_config` 外置，是因为目标指标未来会动态变化。

第一版应支持基于结构化配置做本地确定性判断，而不是把评测策略写进模型 Prompt。

## Evaluation Result

`server2` 返回的评测结果应保持稳定、结构化。

建议字段：

- `job_id`
- `candidate_id`
- `dataset_fingerprint`
- `cache_hit`
- `tp_stats`
- `tn_stats`
- `merged_metrics`
- `failure_samples`
- `raw_artifacts`

`failure_samples` 至少要区分：

- `tp.jsonl` 中的漏检样本
- `tn.jsonl` 中的误报样本

## Server2 API

第一版服务契约应采用异步任务语义。

### `POST /eval/jobs`

提交新的评测任务。

输入应包含：

- `candidate_id`
- `system_prompt`
- `user_template`
- `tp_dataset`
- `tn_dataset`
- `metric_config`

响应应包含：

- `job_id`
- `status`
- 如果命中缓存，可附带结果摘要

`status` 至少应支持：

- `queued`
- `running`
- `cached`
- `completed`
- `failed`

### `GET /eval/jobs/{job_id}`

返回当前任务状态，以及在可用时返回最终结果。

响应应包含：

- `status`
- `progress`
- `result` 或 `error`

### 缓存查询边界

第一版不需要单独暴露缓存查询接口。缓存解析应内聚在 `POST /eval/jobs` 内部完成。

只有在后续运维或性能需求明确存在时，再考虑补充单独的缓存查询 API。

## 缓存设计

第一版必须实现结果缓存。

缓存键必须覆盖所有语义上会影响评测结果的输入：

- `system_prompt`
- `user_template`
- `tp.jsonl` 内容哈希
- `tn.jsonl` 内容哈希
- `server2` 上评测模型的版本
- 相关推理参数

这样可以避免在 Prompt、数据集或运行环境变化时出现错误缓存命中。

## 优化闭环

本地控制器执行一个有边界的迭代循环。

### 循环步骤

1. 加载当前最佳 Candidate
2. 将最佳版本、最近历史、指标和代表性失败样本发送给 `server1`
3. 从 `server1` 收到修订后的新 Candidate
4. 将新 Candidate 提交给 `server2`
5. 轮询直到任务完成
6. 在本地根据当前 best 和运行时指标策略做判断
7. 如果 challenger 获胜，则将其晋级
8. 持续迭代，直到命中停止条件

### 发给 `server1` 的反馈内容

每轮至少应包含：

- 当前最佳 Candidate
- 最近几轮的修改历史
- 最新一轮的汇总指标
- 代表性的 `tp` 失败样本
- 代表性的 `tn` 失败样本

本地控制器应限制失败样本数量，只反馈一小批具有代表性的样本。若把所有失败样本全部传回给 `server1`，会产生过多噪声，导致修订不稳定。

### `server1` 必须输出的候选信息

本地控制器至少应要求 `server1` 返回：

- 修订后的 `system_prompt`
- 修订后的 `user_template`
- 简明的修改理由
- 对本次修改预期效果的说明

这些说明对于可追踪性和后续调试是必须的。

## Judge

晋级决策必须是本地的、确定性的。

`server1` 可以提出候选修改，但不能决定候选是否成功。该判断由本地控制器中的 `Judge` 组件负责。

`Judge` 应支持：

- 根据 `metric_config` 检查硬约束
- 将 challenger 与当前 best 进行比较
- 返回明确的决策及原因

第一版采用简单的单 best 策略：

- 一个 `best` Candidate
- 一个当前待测的 challenger

这样可以避免不必要的复杂度，同时仍然实现完整闭环。

## 停止条件

当任一配置的停止条件被触发时，循环结束。

第一版至少支持：

- 满足目标指标
- 达到最大轮数
- 连续 N 轮无有效提升
- `server1` 生成重复或高度相似的修改
- 连续缓存命中且没有产生真正的新候选

## 错误处理

第一版必须明确处理三段链路中的主要失败模式。

### Server1 失败

- 对瞬时请求失败进行有限重试
- 生成失败时不能覆盖当前 best
- 在本地保留失败尝试记录

### Server2 失败

- 将任务标记为 `failed`
- 保留提交输入，便于重放
- 允许本地控制器选择重试或跳过

### 非法结果

- 拒绝结构不合法的结果载荷
- 拒绝样本数不一致的结果
- 阻止非法结果进入 Judge 阶段

### 缓存安全

- 不能只按 `candidate_id` 做缓存
- 必须基于内容指纹做缓存

## 测试策略

测试应分阶段进行。

### 单元测试

覆盖：

- Judge 决策逻辑
- 缓存指纹生成
- 失败样本抽样
- 结果聚合和 schema 校验

### 集成测试

通过 mock `server1` 和 `server2` 验证：

- 闭环流程推进
- 晋级与淘汰逻辑
- 重试行为
- 停止条件行为

### Smoke Test

使用一小份真实数据集运行：

- 一版 baseline candidate
- 一到两版演化后的 candidate

该 smoke test 应验证：

- 任务提交
- 状态轮询
- 结果缓存
- 反馈构造
- 本地实验记录落盘

## 实现顺序

第一版建议按以下顺序落地：

1. 将 `server2/batch_run.py` 改造成评测服务内核
2. 在 `server2` 上增加异步任务处理与缓存
3. 构建本地控制器，使其可以提交一次评测并读取一次结果
4. 将本地控制器接入 `server1`
5. 实现确定性的 Judge 和有边界的多轮循环
6. 扩展 Candidate schema，为未来参数搜索预留字段，但第一版先不启用

## 第一版明确不做的内容

以下内容明确不纳入第一版：

- 多候选并行分支
- 搜索树式优化
- `server2` 上的自动策略生成
- 学习型 reward model
- 本地大模型部署
- 除 Prompt 与模板外的大规模超参数搜索

## 预期结果

第一版完成后，项目应能够：

- 在本地运行完整的优化控制器
- 使用 `server1` 作为远端大模型大脑
- 使用 `server2` 作为带缓存的异步评测后端
- 通过基于证据的多轮修订，持续提升 Prompt Attack Detection 效果
- 对每一版 Candidate 的修改内容、修改原因和评测结果保留可审计记录
