# Wikidata CTA 数据集构造协议

本文记录 DART CTA retrieval 数据集 `cta_retrieval_dataset_v9_1_full`
的构造、ontology artifact、数据划分和验证方法。目标是保留论文写作与未来
benchmark 重建所需的可复现信息。

## 1. 数据集定位

数据集用于训练一个 recall-oriented column type retriever：

- 输入：列标题和若干文本型单元格；
- 输出：Wikidata ontology type 候选；
- 训练目标：使一个或多个可接受类型进入前 K 个候选；
- 后续阶段：LLM reranker 在候选集合中完成精排。

当前数据只覆盖 entity-valued columns。数值、日期、单位等 literal columns
不属于该版本的标注范围。

## 2. 固定版本与数据来源

| 项目 | 固定值 |
|---|---|
| Wikipedia | English Wikipedia，snapshot `2025-03-20` |
| Wikidata | snapshot `2025-03-20` |
| 表格来源 | `kgdata` 的 `linked_relational_tables` |
| 实体类型 | Wikidata `instance of (P31)` |
| 类型层级 | Wikidata `subclass of (P279)` |
| 构建随机种子 | `42` |
| 输出 schema | `2` |

HPC workspace 中的主要输入：

```text
$WS/wikipedia/20250320/linked_relational_tables/
$WS/wikidata/20250320/
$WS/wikidata_db/entities.db
$WS/wikidata_db/entity_labels.db
$WS/wikidata_db/classes.db
```

构建过程完全离线，不依赖 Wikidata SPARQL endpoint。这样可以固定知识库版本，
避免在线结果变化、限流和查询失败。

## 3. 必须归档的生成代码

v9.1 实际由 HPC `$WS/code` 中的以下文件生成：

```text
kgdata_build_dataset.py
cta_labeling.py
table_safety.py
```

这些文件必须与数据 artifact 一起归档，并记录 Git commit 或独立 SHA-256。

> **当前复现风险**
>
> 仓库中的
> `WIKIDATA/build_traning_data/kgdata_build_dataset.py`
> 仍是较早的 single-positive 实现，与 v9.1 schema 和实际 HPC 生成代码不完全
> 一致。重新发布数据或构造 benchmark 前，必须把上述 HPC 版本同步回仓库。
> 在完成同步前，不应声称仅凭当前仓库 checkout 可以逐字节重建 v9.1。

## 4. 原始表格读取

### 4.1 输入记录

`linked_relational_tables` 是 gzip JSONL shards。每条记录包含：

- `LinkedHTMLTable.table`：解析后的 `rsoup.Table`；
- `LinkedHTMLTable.links`：单元格坐标到 `WikiLink` 列表的映射；
- Wikipedia page URL；
- Wikipedia link 对应的 Wikidata QID。

本次运行共发现 497 个 `linked_relational_tables` shard files。

### 4.2 表格标识

每张表使用包含 `table_no` 的 URL 标识：

```text
https://en.wikipedia.org/wiki/<PAGE>?table_no=<N>
```

该字段写入 `table_id`，并与 `col_index` 一起唯一追踪来源列。

### 4.3 列提取

对每张 relational table：

1. 读取表格形状；
2. 将第 0 行作为 header row；
3. 对每一列读取 header；
4. 遍历数据行；
5. 从单元格链接中选择主实体 QID；
6. 保存可见单元格文本和实体 QID；
7. 链接单元格少于阈值的列不进入类型推断。

配置：

```text
min_linked_cells = 3
max_cell_samples = 10
```

`min_linked_cells` 在 entity 去重之前判断。因此，重复链接同一实体的列可能满足
3 个 linked cells，但 `num_typed_entities` 小于 3。当前完整数据中
`num_typed_entities` 的实际最小值为 1。未来 benchmark 可考虑同时要求最小
unique-entity count。

## 5. RichText 与链接安全处理

Wikipedia 单元格可能包含 flag icons、引用、模板、图片和损坏的 RichText
offsets。v9.1 采用以下行为：

- header 和 cell value 使用 `RichText.text`，不使用 HTML serialization；
- 输出文本不得包含 `<th>`、`<td>`、`<a>` 等 HTML；
- 优先选择覆盖可见文本的语义链接；
- 不把零长度 flag/icon 链接当作主实体；
- 引用链接和装饰性链接不应覆盖真正的实体链接；
- RichText offset 异常时安全跳过单元格或列，不终止全量构建；
- 单元格文本去除首尾空白；
- cell samples 先去重，再最多采样 10 个。

该修复解决了早期版本中 `"Team"` 列被 flag 对应的 football federation
错误标注，而不是 football club 的问题。

## 6. 基于 P31 的列类型推断

### 6.1 实体级计数

同一列中的实体 QID 先去重。每个 unique entity 对一个 direct P31 type
最多贡献一次支持。

设：

- \(E\)：至少具有一个有效 P31 的 unique entities；
- \(E_t\)：direct P31 包含类型 \(t\) 的 entities。

类型覆盖率为：

```text
coverage(t) = |E_t| / |E|
```

一个实体可以具有多个 P31，因此多个类型的覆盖率之和可以大于 1。

记录中的证据字段：

```text
type_entity_counts
num_typed_entities
positive_type_support
```

### 6.2 Primary positive

候选类型按以下顺序排序：

1. entity support count 降序；
2. QID 升序，作为确定性 tie-break。

支持最高的类型为 `positive_type_qid`。只有其覆盖率达到以下阈值时才接受该列：

```text
majority_threshold = 0.7
```

`majority_ratio` 保存 primary type 的 entity coverage。

### 6.3 Multi-positive labels

除了 primary positive，其他 direct P31 type 在覆盖率达到以下阈值时也作为
有效 positive：

```text
positive_coverage_threshold = 0.5
```

所有有效 positives 写入：

```text
positive_type_qids
positive_type_labels
positive_type_support
```

`positive_type_qid` 和 `positive_type_label` 保留，用于兼容旧版 reader。

完整数据的 positive-set size：

| positives/record | records |
|---:|---:|
| 1 | 124,741 |
| 2 | 30,488 |
| 3 | 9,555 |
| 4 | 4,099 |
| 5 | 2,121 |
| 6 | 973 |
| 7 | 816 |
| 8 | 366 |
| 9 | 288 |
| 10 | 132 |
| 11–20 | 397 |

共 49,235 条记录具有多个 positives，占 28.30%。

## 7. 结构和语义噪声过滤

Wikipedia relational tables 仍包含导航表、评分表、交易表和弱语义列。
v7–v9.1 的人工抽样检查驱动了以下过滤：

- 空 header；
- 只有结构意义、不能稳定表示实体类型的列；
- navigation/template 内容；
- reference、notes、attendance、signed 等弱语义字段；
- review-score 等表格布局字段；
- header 与推断类型明显冲突的记录；
- 由装饰链接或错误 primary link 造成的类型偏差。

过滤原则：

1. 只在可明确证明列不适合 CTA supervision 时删除；
2. 不用少量 header blacklist 替代语义判断；
3. 每轮修改后同时检查已知错误案例和随机语义样本；
4. 记录过滤前后实例数和类型数。

在 5,000-table smoke runs 中，记录数随清洗增强而减少：

```text
v6:   1,419 records, 369 primary types
v7:   1,114 records, 298 primary types
v8:   1,040 records, 292 primary types
v9:   1,010 records, 288 primary types
v9.1:   997 records, 285 primary types
```

这些数字用于观察清洗影响，不是完整数据统计，也不能直接解释为精度。

## 8. Hierarchy-aware exclusion

对每条记录，从所有 accepted direct positives 出发沿 P279 向上遍历：

```text
hierarchy_ignore_depth = 2
```

以下 QIDs 加入 `ignored_type_qids`：

- 所有 direct positives；
- 指定深度内的 P279 ancestors；
- 其他被判断为语义兼容、不能安全作为 negative 的相关类型。

这些 ancestors 不被提升为等权 positives。原因是：

- direct type 通常比高层 ancestor 更具体；
- unseen ontology 可能采用不同抽象层级；
- 将 ancestor 当作强 negative 会制造 false-negative supervision；
- 将所有 ancestors 当作等权 positive 又会鼓励过度泛化。

因此，当前策略是“direct types 为 positives，合理 ancestors 只排除为
negatives”。

## 9. Negative 构造

每条记录输出：

```text
num_hard_negatives = 5
```

静态 negative 候选按以下优先级生成：

1. 与 primary type 共享 parent 的 sibling types；
2. P279 图中的 cousin types；
3. 当层级候选不足时，从全局类型池随机补充。

每条记录独立应用 exclusion set：

- 不得包含任何 `positive_type_qids`；
- 不得包含任何 `ignored_type_qids`；
- 不得包含重复 QID；
- QID 与 label 列表必须等长。

当前训练日志中的：

```text
0 with mined hard negatives
156662 with original
```

表示模型训练使用数据构建阶段生成的静态 negatives，没有使用训练后模型重新
检索得到的 model-mined negatives。论文中应称为
`hierarchy-aware static negatives`，不能称为 model-mined hard negatives。

## 10. 类型频率控制

构建参数：

```text
max_examples_per_type = 200
```

对每个 primary type 最多保留 200 条记录。超过上限时使用固定 seed 抽样。
该操作降低高频类型对训练的支配，同时保留 long-tail types。

完整输出中 primary type frequency：

```text
minimum = 1
maximum = 200
```

该上限不会使类型完全均衡。未来 benchmark 应分别报告 micro、macro 和
frequency-bucket metrics。

## 11. Schema v2

每行是一个 JSON object：

| 字段 | 类型 | 含义 |
|---|---|---|
| `anchor_header` | string | 清洗后的列标题 |
| `anchor_cells` | list[string] | 去重后的单元格样本，最多 10 个 |
| `positive_type_qid` | string | primary direct P31 type |
| `positive_type_label` | string | primary type English label |
| `positive_type_qids` | list[string] | 所有 accepted direct positives |
| `positive_type_labels` | list[string] | 与 positive QIDs 平行 |
| `positive_type_support` | object | positive QID 到 entity coverage |
| `ignored_type_qids` | list[string] | 禁止作为 explicit negatives 的类型 |
| `type_entity_counts` | object | direct P31 type 的 unique-entity count |
| `num_typed_entities` | integer | coverage denominator |
| `hard_negative_type_qids` | list[string] | 5 个静态 negatives |
| `hard_negative_type_labels` | list[string] | 与 negative QIDs 平行 |
| `table_id` | string | Wikipedia table URL |
| `col_index` | integer | 原表列号 |
| `majority_ratio` | number | primary type coverage |

必要不变量：

```text
positive_type_qid ∈ positive_type_qids
positive_type_qids ∩ hard_negative_type_qids = ∅
ignored_type_qids ∩ hard_negative_type_qids = ∅
len(positive_type_qids) = len(positive_type_labels)
len(hard_negative_type_qids) = len(hard_negative_type_labels) = 5
0.0 <= support <= 1.0
```

## 12. 完整构建配置与输出

```json
{
  "schema_version": 2,
  "total_examples": 173976,
  "unique_types": 7724,
  "label_not_found": 20,
  "num_shards": 4,
  "config": {
    "num_hard_negatives": 5,
    "max_cell_samples": 10,
    "min_linked_cells": 3,
    "majority_threshold": 0.7,
    "positive_coverage_threshold": 0.5,
    "hierarchy_ignore_depth": 2,
    "max_examples_per_type": 200,
    "max_tables": null,
    "seed": 42
  }
}
```

注意：

- metadata 中的 `unique_types=7,724` 是进入 capped type pool 的 primary
  types；
- 其中 20 个缺少可用 label，未写入训练 records；
- JSONL 中实际出现 7,704 个 primary positive types；
- 所有 positive 集合的并集包含 9,725 个 QIDs；
- positives 与 negatives 的候选并集包含 9,745 个 QIDs。

完整数据统计：

| 项目 | 数值 |
|---|---:|
| Records | 173,976 |
| Source pages | 112,842 |
| Source tables | 156,224 |
| Distinct headers | 19,703 |
| Primary positive types in JSONL | 7,704 |
| All positive types | 9,725 |
| Ontology candidate types | 9,745 |
| Multi-positive records | 49,235 |

输出 shards：

```text
train_shard_0000.jsonl
train_shard_0001.jsonl
train_shard_0002.jsonl
train_shard_0003.jsonl
metadata.json
```

SHA-256：

```text
b75d7b78164dba35e88a1c7c0d881189032c6e2b5a6a9c932301fe3d0a0530c4  train_shard_0000.jsonl
d0d63223450beebfffb46d2741b949c9086b23e9f99b87ae2a10ab58ee81fb86  train_shard_0001.jsonl
d7d7af854d80cab09004341372431dc17f03a2980d9faab784895c091f5db51d0  train_shard_0002.jsonl
b01ac36e6c6e1427c92961d324e35b1d60b40f01bc190cea913296de658b1846  train_shard_0003.jsonl
```

## 13. Ontology artifact

### 13.1 Candidate universe

Ontology candidate universe 是所有训练 records 中以下 QIDs 的并集：

```text
positive_type_qids ∪ hard_negative_type_qids
```

总计 9,745 个候选类型。

### 13.2 Ontology term representation

每个候选类型包含：

```text
qid
label
description
aliases
parents
ancestors
children
examples
```

构造规则：

- label、description 和 aliases 来自固定 Wikidata RocksDB snapshot；
- aliases 去重，并删除与 primary label 相同的值；
- `parents` 是 direct P279 parents，包含 QID 和 label；
- `ancestors` 采用 cycle-safe breadth-first traversal；
- maximum ancestor depth 为 32；
- `children` 只包含 candidate universe 内的 direct children；
- `examples` 只从该类型作为 positive 的列中提取；
- 每个类型最多保存 10 个去重 cell examples；
- 缺失 label 时回退到 QID；
- 缺失 description、aliases 或 class metadata 时保留空值并写入 missing report。

### 13.3 Artifact 统计

| 项目 | 数值 |
|---|---:|
| Ontology records | 9,745 |
| 有 description | 8,857 |
| 有 aliases | 6,311 |
| 有 direct parents | 9,603 |
| 有 ancestors | 9,603 |
| 有 candidate children | 2,277 |
| 有 positive examples | 9,725 |
| 至少缺一个可选元数据字段 | 3,704 |

`missing_record_count=3,704` 不表示 3,704 个候选完全缺失。它表示这些 records
至少缺少 entity、label、description、aliases 或 class 中的一项。

输出：

```text
wikidata_type_ontology.jsonl
wikidata_type_ontology.meta.json
wikidata_type_ontology.missing.jsonl
```

独立 validator 检查：

- training candidate 是否全部存在；
- 是否出现额外 candidate；
- QID 是否合法、唯一且排序；
- required fields 类型是否正确；
- parent object 是否包含 string QID 和 label；
- examples 是否超过上限。

当前验证结果：

```text
training_record_count = 173976
training_candidate_count = 9745
ontology_record_count = 9745
violations = 0
```

## 14. Page-disjoint train/validation split

### 14.1 为什么不能随机拆 records

同一 Wikipedia page 中的多个表格通常共享主题、实体、列标题和模板。随机拆列会
使近重复结构同时进入 train 和 validation，产生过高估计。

### 14.2 Page key

从 `table_id` 去掉 `?table_no=<N>`，得到 Wikipedia page key。同一 page
的所有 tables 和 columns 必须进入同一 partition。

### 14.3 确定性分配

```text
assignment = SHA256(seed:page_key) modulo ratio denominator
seed = 42
requested validation ratio = 0.1
```

这种分配不依赖输入顺序，重新分片不会改变 page assignment。

### 14.4 Split 统计

| 项目 | Train | Validation |
|---|---:|---:|
| Records | 156,662 | 17,314 |
| Pages | 101,557 | 11,285 |
| Tables | 140,816 | 15,408 |
| Primary types | 7,428 | 3,005 |
| All positive types | 9,370 | 3,829 |

其他统计：

```text
actual validation ratio = 0.09951947395042994
page intersection = 0
table intersection = 0
validation seen positive types = 3474
validation unseen positive types = 355
malformed table IDs = 0
```

输出：

```text
splits/page_90_10_seed42/
├── train/
│   └── train_shard_*.jsonl
├── val/
│   └── train_shard_*.jsonl
└── split_metadata.json
```

Validation 中的 355 个 train-unseen positive types 适合单独分析，但当前
validation 仍用于 checkpoint selection，不能替代最终 unseen-ontology test。

## 15. 质量验证

### 15.1 自动结构验证

全量数据至少检查：

- 所有 JSONL 行可解析；
- required fields 存在且类型正确；
- positive/ignored 与 negatives 无交集；
- QID 与 label list 长度一致；
- 每条记录恰好有 5 个 negatives；
- header 和 cells 不含 HTML；
- labels 无异常外层引号；
- train/validation records 数量守恒；
- page overlap 和 table overlap 均为 0；
- ontology candidate coverage 为 100%。

### 15.2 Targeted regression cases

每次更改清洗逻辑后检查已知错误案例，例如：

- FIFA Club World Cup 的 `Team` 列必须链接到 football clubs，而非 flags
  对应的 federations；
- `Confed.` 列允许 sport governing body 等合理 positives；
- `Signed`、`Review scores` 等结构列不能产生已知错误 records；
- 损坏 RichText offsets 不能终止构建。

### 15.3 人工语义抽样

每个版本随机抽取 50–100 条，展示：

```text
HEADER
CELLS
POSITIVES
NEGATIVES
TABLE
```

人工判断：

- cells 是否构成一致的实体集合；
- primary type 是否合理；
- additional positives 是否语义兼容；
- negatives 是否具有区分性；
- 是否存在表格结构、引用或模板噪声。

人工抽样用于发现错误模式，不能作为正式精度估计。未来 benchmark 需要独立、
分层、多人标注。

## 16. 当前局限

1. 标签来自 Wikidata P31 distant supervision，不是人工 gold labels。
2. Wikipedia hyperlinks 不完整，未链接实体会被忽略。
3. 一个 cell 的 primary link 选择仍可能错误。
4. P31 粒度不统一，同列可能同时出现具体类和宽泛类。
5. 过滤规则由错误案例驱动，可能误删困难但有效的列。
6. 数据仅来自 English Wikipedia，存在语言和主题偏差。
7. 高频类型虽被 capped，类型分布仍然长尾。
8. 当前只处理 entity columns，不覆盖 literal semantics。
9. 静态 negatives 不等于模型检索到的 hardest negatives。
10. validation 用于模型选择，不能作为最终 benchmark test。
11. 当前仓库尚未归档实际 v9.1 HPC builder，代码级复现仍有缺口。

## 17. 升级为 benchmark 的要求

未来 benchmark 至少需要补充：

### 数据冻结

- 固定 Wikipedia/Wikidata dump checksums；
- 固定 kgdata、rsoup、Python 和 RocksDB 版本；
- 归档实际 builder、配置、日志和 Git commit；
- 对所有 shards 和 ontology artifacts 发布 checksums。

### Gold annotation

- 从自动数据中分层抽取独立 test set；
- 至少两名 annotators；
- 明确定义 direct、acceptable ancestor 和 incorrect type；
- 保存 disagreements 和 adjudication；
- 报告 inter-annotator agreement；
- 不把自动 P31 majority label 直接称为 gold。

### 防泄漏划分

- 至少保持 page-disjoint；
- 检查实体重叠和近重复列；
- 对模板化表格考虑 template-disjoint split；
- 对 unseen-type、unseen-domain 和 unseen-ontology 分别建 test slice。

### 指标

- Recall@1、Recall@5、Recall@10、Recall@20；
- micro 和 macro Recall；
- seen/unseen type；
- head/medium/tail frequency buckets；
- direct-type 与 acceptable-ancestor 分层结果；
- candidate coverage 与 LLM reranker end-to-end accuracy 分开报告。

### 必要消融

- single-positive vs. multi-positive；
- 无 hierarchy exclusion vs. hierarchy exclusion；
- random vs. sibling/cousin negatives；
- record-random vs. page-disjoint split；
- label-only vs. label+description+parents ontology representation；
- 有无 examples；
- 旧数据 vs. v9.1 数据。

## 18. HPC 复现命令模板

实际执行前，先将已归档的 v9.1 builder 放入仓库并锁定 commit。构建参数必须保持：

```bash
python kgdata_build_dataset.py \
    --wp-dir "$WS/wikipedia/20250320" \
    --wd-dir "$WS/wikidata/20250320" \
    --wd-db "$WS/wikidata_db" \
    --out-dir "$WS/cta_retrieval_dataset_v9_1_full" \
    --num-hard-negatives 5 \
    --max-cell-samples 10 \
    --min-linked-cells 3 \
    --majority-threshold 0.7 \
    --positive-coverage-threshold 0.5 \
    --hierarchy-ignore-depth 2 \
    --max-examples-per-type 200 \
    --seed 42
```

构建 ontology artifact：

```bash
python -m WIKIDATA.ontology_parser.build_wikidata_artifact \
    --train-path "$WS/cta_retrieval_dataset_v9_1_full" \
    --wd-db "$WS/wikidata_db" \
    --output \
      "$WS/cta_retrieval_dataset_v9_1_full/wikidata_type_ontology.jsonl" \
    --max-examples 10 \
    --max-ancestor-depth 32
```

独立验证：

```bash
python -m WIKIDATA.ontology_parser.validate_wikidata_artifact \
    --train-path "$WS/cta_retrieval_dataset_v9_1_full" \
    --ontology \
      "$WS/cta_retrieval_dataset_v9_1_full/wikidata_type_ontology.jsonl" \
    --max-examples 10
```

创建 page-disjoint split：

```bash
python data/split_training_data.py \
    --input-dir "$WS/cta_retrieval_dataset_v9_1_full" \
    --output-dir "$WS/cta_retrieval_dataset_v9_1_split_page_90_10_seed42" \
    --val-ratio 0.1 \
    --seed 42
```

## 19. 训练结果附录

以下结果只用于确认数据可训练和选择 checkpoint，不是最终 benchmark test：

| Epoch | Recall@1 | Recall@5 | Recall@20 |
|---:|---:|---:|---:|
| 1 | 0.4840 | 0.7759 | 0.9060 |
| 6 | 0.6263 | 0.8706 | 0.9496 |

训练设置：

```text
encoder = intfloat/multilingual-e5-base
epochs = 6
batch_size = 16
learning_rate = 2e-5
max_length = 192
max_cells = 10
max_parents = 3
mask_header_probability = 0.2
gradient_checkpointing = enabled
```

最佳 checkpoint：

```text
output/dart_1662057/best_model
```

正式论文结果还需固定随机种子、运行多次，并在独立 unseen ontology test set
上比较旧数据、v9.1 和各项消融。
