# 个人学术主页改版调研

调研日期：2026-10-07。研究对象：`huang2202/huang2202.github.io`。

## 目标与研究方法

用户希望个人页面更接近国际一流研究者的表达方式，降低当前页面的娱乐感和学生式自我介绍。本阶段完成第一方网站调研、现有网站审阅和设计讨论准备；以下改版方案均为建议，尚未成为实施决定。

样本围绕具身智能、机器人学习、强化学习与计算机视觉选取，并加入理论研究者、跨地区学者和早期研究者作为对照。这是有目的的案例比较，不是学者、实验室或网站设计的权威排名。研究质量也不能由网页外观推断。

有效样本涵盖 18 位研究者（其中 1 位为早期研究者对照）、6 个实验室，并以 MIT CSAIL 作机构级对照。学者样本包括独立个人站、机构域名下的个人页、个人简历/出版目录和机构人员档案；它们的职责在专题笔记中分别标注。没有读到有效正文的站点不计作独立完整样本。

研究材料分为三种：

- 第一方网站的文字、栏目和资源链接，用于核实内容组织。
- 实际 Chrome 渲染的 1440 × 1100 桌面首屏截图，用于核实可见布局。
- 本地源码审阅，用于识别当前页面内容、模板残留和数据状态。

只通过文本阅读的样本不用于判断字体、配色、动效或移动端表现。截图是一次桌面视口快照，不是完整交互、性能或无障碍审计。

## 专题笔记

- [AI 与机器人学者个人页](./scholar-pages-ai.md)
- [跨地区学者个人页](./scholar-pages-global.md)
- [研究实验室与机构网站](./lab-pages.md)

## 重点视觉样本

以下截图已实际查看，来源均为对应网站。观察仅限截图中可见的首屏。

| 样本 | 可核实的首屏结构 | 可借鉴之处 | 快照 |
| --- | --- | --- | --- |
| [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/) | 白底、衬线正文、姓名与肖像；机构、邮箱、研究概述、实验室与论文入口；随后为研究演讲 | 用一段研究概述连接身份与工作资源 | [截图](./screenshots/2026-10-07/levine.png) |
| [Chelsea Finn](https://ai.stanford.edu/~cbfinn/) | 白底；简介与肖像并列；正文强调研究兴趣，集中列出 CV、论文资料和实验室链接；下方新闻与演讲 | 首屏用明确的研究兴趣和简短资源导航建立学术语境 | [截图](./screenshots/2026-10-07/finn.png) |
| [Shuran Song](https://shurans.github.io/) | 白底、蓝色姓名和链接；照片与身份、研究议程并列；下方讲座与论文栏目 | 个人简介、所属实验室和具体研究目标放在同一区域 | [截图](./screenshots/2026-10-07/song.png) |
| [Jon Barron](https://jonbarron.info/) | 白底、集中内容列；简介与肖像；Research 概述后紧接图文论文列表，条目包含作者、年份、资源和一句结果说明 | 精选成果的缩略图承担研究信息，列表能直接通向论文和项目 | [截图](./screenshots/2026-10-07/barron.png) |
| [Chunhua Shen](https://cshen.github.io/) | 白底文本；Home、Publications、Teaching；中英文身份简介、研究议程、新闻与学生指导 | 跨语言身份信息可以简洁并存，研究说明比方向关键词更具体 | [截图](./screenshots/2026-10-07/shen.png) |
| [Berkeley RAIL](https://rail.eecs.berkeley.edu/) | 简短导航、大幅户外团队合照、研究目标句；入口为成员、论文、软件和联系 | 学术网站可以有个人或团队温度；团队结构服务实验室用途 | [截图](./screenshots/2026-10-07/rail.png) |
| [Stanford IRIS](https://irislab.stanford.edu/) | 顶部简短导航、大幅团队合照、研究目标与机构关系 | 借鉴研究目标表达；成员和团队视觉不必移植为个人主页结构 | [截图](./screenshots/2026-10-07/iris.png) |

这些截图呈现了多种学术网站样式：默认 HTML 式的文本页、留白较多的个人资料页、图文研究作品列表和团队首页。共同的内容线索是身份、研究问题、工作资源之间的关系。浅色低对比正文、大量首屏空白或过时的日期也可能出现，不能因为作者知名就把所有视觉细节都视为最佳实践。

## 补充个人页样本

这些样本直接阅读了第一方主页正文与链接，未做视觉风格判定。

| 样本 | 直接观察到的内容策略 | 对本项目的启发 |
| --- | --- | --- |
| [Dhruv Batra](https://dhruvbatra.com/) | 导航区分论文、Essays、Teaching 和 Academic Lab；当前与过去身份分开；代表项目按研究主题组织 | 写清现在做什么，把研究写作与完整论文目录分别组织 |
| [Dieter Fox](https://homes.cs.washington.edu/~fox/) | 身份和研究组之后用独立段落解释研究目标、感知与控制关系，再列成果与教学资源 | 用研究问题和方法关系代替纯关键词清单 |
| [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/) | 姓名、机构、邮箱之后直接列 Bio、CV、Scholar、课程、演讲与历史研究入口 | 首屏作为学术资源入口，适合信息量较大且长期维护的档案 |
| [Scott Aaronson](https://www.scottaaronson.com/) | 身份与研究主题后列 Research、Blog、CV、讲义、课程及其他写作 | 博客和个人声音可以保留，内容分类让读者选择需要的材料 |
| [Saining Xie](https://www.sainingxie.com/) | 简介解释研究主张，随后列研究组和精选论文；论文提供项目、文章与代码等资源 | 让长期研究主张通过具体作品获得支撑 |
| [Yanjie Ze](https://yanjieze.com/)（早期研究者对照） | 明确写 PhD student；简短研究主张，随后为 Highlights、带日期的 News、论文与研究工具 | 学生身份与专业表达可以并存；真实成果和明确方向是内容基础 |

完整地区比较见[跨地区笔记](./scholar-pages-global.md)，包括 Andrew Zisserman、Yoshua Bengio、Judea Pearl、Masashi Sugiyama 与 Chunhua Shen。实验室比较见[实验室笔记](./lab-pages.md)，包括 RAIL、IRIS、SVL、BAIR、Oxford VGG、MIT Improbable AI，并以 MIT CSAIL 作机构级对照。不同站点和子页的访问限制在各笔记中记录。

## 综合判断

以下是基于样本比较形成的设计推论，而非各来源自行声明的统一规则。

1. **学术可信度由内容路径支撑。**读者能从准确身份进入具体研究问题，再进入论文、代码、演讲或项目资料；CV 和 Scholar 提供进一步核验入口。[Levine](https://people.eecs.berkeley.edu/~svlevine/) · [Finn](https://ai.stanford.edu/~cbfinn/) · [Barron](https://jonbarron.info/)
2. **成熟研究者会解释研究议程。**方向词之后还有研究目标、方法关系或研究转向的说明；这使多项工作呈现为有联系的研究脉络。[Fox](https://homes.cs.washington.edu/~fox/) · [Bengio Research](https://yoshuabengio.org/en/research) · [Xie](https://www.sainingxie.com/)
3. **成果条目提供可操作的资源入口。**标题、作者、年份与发表状态之外，项目页、论文和代码让读者继续阅读或复现。研究视频应说明任务与结果，图示应解释研究内容。[Song](https://shurans.github.io/) · [IRIS Publications](https://irislab.stanford.edu/publications.html) · [VGG Publications](https://www.robots.ox.ac.uk/~vgg/publications/)
4. **个人声音可以与研究内容并存。**博客、Essays、讲义或团队生活照片在多个样本中出现。应通过内容层级处理个人兴趣与研究叙事的关系。[Batra](https://dhruvbatra.com/) · [Aaronson](https://www.scottaaronson.com/) · [RAIL](https://rail.eecs.berkeley.edu/)
5. **早期研究者需要准确表达阶段和公开贡献。**少量真实成果、公开代码、可复现研究项目或高质量研究写作足以形成清晰页面；栏目名称和状态应与内容相符。[Ze](https://yanjieze.com/) · [Sugiyama Software](https://www.ms.k.u-tokyo.ac.jp/sugi/software.html)

## 当前网站基线

[线上首屏快照](./screenshots/2026-10-07/current-live.png)由 Chrome 实际加载取得。文本浏览工具未能打开该域名，但浏览器截图成功；因此不把文本抓取失败解释为部署故障。线上快照未核对部署 commit，代码结论以本地仓库为准。

| 观察 | 本地依据 | 改版意义 |
| --- | --- | --- |
| Lumin Space、Harry-Guang、Journey to the North Star 和动漫头像形成当前首屏身份 | [站点配置](../../src/site.config.ts)、[头像](../../src/assets/nova.png) | 需决定首页是否以学术署名、专业肖像和研究说明为主要识别信息 |
| 首页顺序为个人卡片、About、Education、Statistics、GitHub Activities | [首页](../../src/pages/index.astro) | 研究方向、研究项目和公开成果可前移；统计区占据较多阅读空间 |
| 首页与 Academic 对本科、博士和进组状态的表述不一致 | [首页](../../src/pages/index.astro)、[Academic](../../src/pages/academic/index.astro) | 正式写文案前需要准确的当前身份和时间线，不能猜测 |
| Publications 包含 Author1、Venue、Year 等占位；Scholar 指向服务首页 | [Academic](../../src/pages/academic/index.astro)、[首页](../../src/pages/index.astro) | 用真实资源入口和准确成果记录建立可核验路径 |
| About 包含占位自述与工具清单；页脚有模板式备案信息，配置中有随机语录 | [About](../../src/pages/about/index.astro)、[站点配置](../../src/site.config.ts) | 整理首屏之外的内容，避免通用模板持续主导个人叙事 |
| 本地有 14 篇研究阅读、课程学习和个人写作材料；审阅到的元数据均为 `state: off` | `src/content/blog/`、博客路由的状态过滤 | 这些材料不等同本人发表成果；公开范围需由本人决定，不能自动发布草稿 |
| 现有 Astro、静态输出、页面与 Markdown 结构可以承载改版 | [项目配置](../../package.json)、[Astro 配置](../../astro.config.ts) | 可以优先改内容、导航、布局与样式，现阶段没有已证实的技术栈迁移需求 |

其他可处理问题包括无效链接、分类 slug 不一致、页面目录与实际章节不一致，以及项目页中主题作者的资源归属。它们属于源码审阅发现，不在本轮研究阶段自动修复或改为公开内容。

## 建议方案，待讨论

建议方向：以 Finn 式的简洁研究简介、Barron 式的研究作品条目为主要参考，配合用户真实的研究问题和成果储备。

建议导航：Home / Research / Writing / CV。具体页面数量和 URL 仍待确定。

建议首页阅读顺序：

1. 学术署名、当前准确身份、简短研究主张、Email / Scholar / GitHub / CV。
2. 已有真实内容时，显示少量带日期的研究动态。
3. 精选研究工作：成果图或任务演示、研究问题、本人贡献、发表或公开状态、有效资源链接。
4. 简短教育与研究经历。
5. 少量精选研究写作；其余材料进入 Writing。

若暂无适合公开的论文，使用与实际内容相符的 Research Projects 或 Research Notes；空的 Publications 栏目无需靠占位条目补齐。英文主页、真人肖像、浅色背景、清楚的正文对比度和少量强调色是可考虑的表达方式，尚未成为用户选择。

站点统计、GitHub 活跃图、随机语录、工具徽章与友链可以从学术首页移出，是否保留在其他页面需要讨论。已有博客路径与草稿状态先作为约束保留，避免把视觉整理变成内容公开或 URL 迁移。

## 讨论的设计树

已明确的目标：降低娱乐感，以更成熟、准确、可核验的学术表达呈现个人研究。

后续用户选择与当前讨论进展记录在[设计讨论文档](../design/academic-homepage.md)；下表保留调研结束时的第一轮问题。

第一轮可独立讨论的决策：

| 决策 | 推荐方向 | 下游问题 |
| --- | --- | --- |
| 主要受众与访问目标 | 国际同行和潜在研究合作者能迅速判断研究匹配度 | 语言、研究说明深度、首页章节优先级、联系入口 |
| 对外身份与品牌 | 学术署名为主，昵称与站名放在次要位置 | 具体署名、当前身份、肖像、浏览器标题与搜索信息 |
| 研究写作与个人内容边界 | 同域保留 Writing，首页以研究工作为主 | 哪些草稿公开、生活内容是否归档、旧路径是否保留 |

这些决策形成后，再讨论具体视觉取向、可公开的作品与贡献、研究说明、栏目命名、维护成本和迁移约束。每次确定的术语才进入 `GLOSSARY.md`；达到难以反转、有真实权衡且需要背景解释的决定才记录 ADR。研究建议不提前登记成已接受的决定。
