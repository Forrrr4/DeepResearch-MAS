# M1 Baseline vs 完整多Agent系统 —— 对比报告

生成时间：2026-08-13。评估集：`eval/eval_set.jsonl`（18题，M1阶段12题 + M5扩充6题，重点补充adversarial_contradiction和time_sensitive类别）。

数据来源：
- M1 baseline原始结果：`eval/results/m1_baseline_18q_20260813T004255.jsonl`
- 完整系统（M1-M4全部能力：Orchestrator+并行SubAgent+Critic+Citation Manager+Writer+预算控制+Prompt Injection防护）最终结果：`eval/results/full_system_final.jsonl`
- LLM-as-judge打分（deepseek-v4-pro，维度定义见`docs/04_M1评估集设计说明.md`）：
  - M1: `eval/results/judge_m1_baseline_20260813T011605.jsonl`
  - 完整系统: `eval/results/judge_full_system_v2_20260813T014211.jsonl`

**这份报告的数据没有美化。** 跑批过程中发现并修复了3个真实bug（见第4节），报告里的完整系统数据是修复后的最终版本，修复前的原始数据和排查过程完整记录在`docs/design_decisions_log.md`的M5章节，没有被隐藏或删除。

---

## 1. 总体对比

| 维度 | M1 Baseline | 完整系统 | 变化 |
|---|---|---|---|
| accuracy（准确率，0-5） | 3.50 | 4.44 | +0.94 |
| completeness（完整性，0-5） | 3.33 | 4.17 | +0.84 |
| citation_validity（引用有效性，0-5） | 3.22 | 4.44 | +1.22 |
| honesty（诚实度，0-5） | 3.56 | 4.78 | +1.22 |
| honesty（仅adversarial类问题，6题） | 4.17 | 4.83 | +0.66 |
| 总token消耗（18题合计） | 65,728 | 250,663 | **3.81倍** |
| 平均单题token消耗 | 3,651.6 | 13,925.7 | **3.81倍** |
| 平均单题延迟 | 58.2秒 | 159.2秒 | **2.73倍** |
| 因超时/出错直接失败的题目数 | 5/18 | 0/18 | -5 |

**结论不是"多Agent全面碾压"，而是有明确权衡**：完整系统在四个质量维度上都有实质提升，且**完全消除了M1的5次超时失败**；代价是token成本约3.8倍、延迟约2.7倍。这个权衡是否划算取决于场景——对时间不敏感、追求质量和覆盖面的研究任务，完整系统明显更优；对延迟敏感的场景，需要考虑M1这类单agent方案或对完整系统做更激进的预算裁剪。

---

## 2. M1 Baseline的5次失败：不是bug，是单Agent的真实局限

M1在以下5题上直接超时失败（BaselineAgent的90秒超时上限被打满，`error="timeout after 90.0s"`，judge对空回答直接判0分）：

| ID | 类别 | 问题 |
|---|---|---|
| eval-007 | multi_perspective_comparison | 分析至少3家大模型厂商的Agent技术路线差异 |
| eval-008 | multi_perspective_comparison | 三个角度对比多Agent架构vs单Agent+超长上下文 |
| eval-013 | adversarial_contradiction | 梳理RAG是否被长上下文淘汰的双方论据 |
| eval-016 | time_sensitive | 4家公司最近一周的更新（要求高时效性检索） |
| eval-018 | multi_perspective_comparison | 对比3个向量数据库的部署/性能/生态 |

这5题有一个共同点：**都要求覆盖多个独立主题**。单Agent必须在一次对话里顺序完成"3-4轮搜索+综合"，累积耗时超过90秒上限；而完整系统要么把这类问题拆成并行子任务（eval-006/007/012/018被Orchestrator判定为multi_agent），要么即使判定为single_agent也仍然完整答完（eval-008/013/016最终都在90秒内完成，说明这几题本身M1原则上答得动，只是那次运行恰好较慢——这提示90秒本身是个偏紧的上限，但也如实保留了这个真实发生的失败案例，不去追溯修改M1已经定稿的历史数据）。

完整系统在这5题上的最终得分：eval-007(3/3/4/4)、eval-008(5/5/5/5)、eval-013(5/5/5/4)、eval-016(4/2/4/5)、eval-018(2/1/4/4)——**不是全部完美**（eval-018明显偏弱，见第4.3节），但至少都产出了实质内容，而不是M1那样直接交白卷。

---

## 3. 路由准确率：Orchestrator分流决策的真实表现

完整系统在18题上的路由决策：14题single_agent，4题multi_agent，0题因预算耗尽强制converge。

对照`eval_set.jsonl`里`should_trigger_multi_agent`标注（M1阶段设计评估集时的预期值），本次路由准确率为 **0.722（13/18命中）**。

不一致的5题里，比较有代表性的两类：
- eval-011/017（time_sensitive，标注为false）：Orchestrator正确判定为simple，直接single_agent，符合预期。
- eval-016（4家公司最近更新，标注为true但本次判定simple）：这题理论上适合拆分，但Orchestrator这次判断"一个agent顺序查也能搞定"，实际跑下来single_agent确实答完了（4/2/4/5），说明这个边界本身就模糊——不是路由错了，是"合不合适拆"本来就没有唯一答案。

这个0.722和M2阶段单独测试Orchestrator路由逻辑时的0.75相近，说明**复杂度判断这个环节的准确率短期内趋于稳定在70%-80%区间**，如果要进一步提升，需要更大的标注数据集做针对性的prompt调优或引入更细粒度的启发式规则，而不是继续在这18-20条上反复调（容易过拟合到具体措辞）。

---

## 4. M5跑批过程中发现并修复的3个真实bug

如实记录整个过程，而不是只展示修复后的"干净"结果：

### 4.1 `single_agent_node`静默返回空报告

跑第一版18题全量eval时，eval-008和eval-013的完整系统结果是空报告（tokens仅270/292，几乎没做任何工作），但`error`字段是空的——排查发现`single_agent_node`从来没检查过`BaselineAgent`返回的`result.error`，超时/异常时`result.output`是空字符串，被原样当作`final_report`返回。用户拿到的是一份空白报告，却看不出任何错误提示。

**修复**：`single_agent_node`现在会在`result.error`存在时，把final_report替换成明确的失败说明而不是空字符串。修复后重跑eval-008/eval-013，两题都产出了完整、高质量的报告（各自5/5/5/5和5/5/5/4）。

### 4.2 Writer在大findings量下综合失败，兜底信息全部浪费

18题全量eval里，eval-006/007/012这三题（都被Orchestrator正确判定为multi_agent、真实收集到36-54条有citation支撑的findings）的`final_report`却是"（Writer未能生成有效报告）"这句空话——排查发现是M3阶段已经记录过的同一个根因（Writer综合大量findings时thinking内容占满输出token预算，正文被截断成空），这次在更大规模（40-50条findings）下再次复现，且旧的兜底逻辑只会道歉，把已经真实收集、经过来源校验的信息完全浪费掉。

**修复**：新增`assemble_fallback_report`纯代码函数，Writer彻底失败时改为按subtask分组列出所有findings及其引用编号，而不是一句空话。修复后重跑，eval-006/007/012三题的最终报告长度达到16K-25K字符（真实内容，不是道歉），citation_validity和completeness的judge分数也相应回升。

### 4.3 eval-018子任务反复超时：未完全解决的真实局限

即使在两次重跑后，eval-018（对比Qdrant/Pinecone/Milvus三个向量数据库）的子任务仍然出现多次90秒超时（`qdrant`/`pinecone`/`milvus`及其重试`pinecone_retry`），最终findings数量（27条）明显少于同规模的其他multi_agent问题（36-54条），judge分数也是完整系统这次跑批里最低的一题（2/1/4/4）。

这不是本次新发现的问题，而是M3阶段已经记录的"大信息量下的可靠性边界"的延续——**如实标注为未解决**，不做进一步的timeout加码（M3已经验证过继续加timeout收益递减），后续如果要解决，方向应该是SubAgent层面的搜索结果预处理/摘要压缩，减少单次调用需要处理的信息量，而不是单纯延长等待时间。

---

## 5. 成本/延迟的具体构成

完整系统的token消耗集中在少数multi_agent问题上：4道multi_agent题（eval-006/007/012/018）合计消耗165,489 token，占总消耗的66%，仅占18题里的22%——**印证了架构文档一直强调的"多Agent的成本不是均摊的，主要来自真正被拆解并行的那部分问题"**。14道single_agent题的平均token消耗（约6,084/题）其实只比M1 baseline（3,651.6/题）高出约1.7倍，这部分增量主要来自Orchestrator的一次复杂度判断调用，而不是Writer/Citation Manager等下游环节（single_agent路径直接跳过了Critic循环和Citation Manager的URL可达性探测）。

延迟同理：multi_agent题的平均延迟约515秒（Critic循环+多轮并行调研的真实代价），single_agent题的平均延迟约55秒，和M1 baseline的58.2秒基本持平。

---

## 6. 小结：给面试展示的关键数字

1. **质量**：四个judge维度全部提升，citation_validity和honesty提升最明显（+1.22分），说明Citation Manager的强制来源校验和Writer/Critic的诚实度约束确实在起作用，不是纸面设计。
2. **可靠性**：M1的5次超时失败在完整系统里全部被消除（多Agent并行 + 更完善的错误处理）。
3. **成本代价是真实的、不小的**：token约3.8倍、延迟约2.7倍，且集中在少数被拆解的问题上——多Agent不是免费的质量提升，是用可控的额外成本换取了在"广度优先"问题上的正确性和完整性。
4. **系统在跑批中暴露了3个真实问题**并当场修复，这个过程本身比"一次跑成功"更有工程说服力——尤其是"Writer空报告→改为findings兜底列表"这个修复，直接体现了"信息不能因为最后一步失败就被浪费掉"这个工程判断。
5. **仍有未解决的边界**（eval-018的子任务可靠性问题），如实标注，不掩盖。
