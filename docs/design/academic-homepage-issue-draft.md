# GitHub Issue 草稿

目标仓库：`huang2202/huang2202.github.io`。

标题：将个人网站整理为研究优先的学术主页（首版）。

提交状态：尚未创建。2026-10-07 通过连接的 GitHub 工具提交时返回 `403: Resource not accessible by integration`。以下正文已经准备完成，可在获得写入权限后直接创建。B 版正式实施及验收已完成，用户已验收预览并明确授权推送部署。

## 正文

当前首页以个人博客式介绍、站点统计和 GitHub 活跃图为主，学术页还包含成果占位。首版改为面向国际研究同行与潜在合作者的英文个人学术主页，突出准确身份、研究兴趣与联系入口。

## 已确定的内容

- 正式姓名：Guangyu Huang；自行选用的公开别名：Harry Guang。
- 姓名区采用 Guangyu Huang (Harry Guang)。
- 机构名称逐字使用：Zhejiang University, State Key Lab of CAD & CG。
- 2026-10-07 首版只列机构；2026-10-13 确认的当前身份见后续需求。
- 研究兴趣以 Robot Learning & Embodied Intelligence 为主线，Reinforcement Learning 作为方法方向。
- 暂无公开研究成果，先建立研究兴趣内容。
- 保留现有动漫头像，使用紧凑尺寸。
- 主页和研究介绍采用英文；原有文章保持原语言。
- 暂时不公开任何笔记。Writing 内容、草稿状态和原有路径保留，首版导航与首页不展示稿件入口或列表。
- 沿用现有邮箱 zju_hgy@163.com 与 GitHub https://github.com/huang2202。

## 首版范围与验收

- [x] 姓名、别名及机构名称与上述资料一致。
- [x] 首页围绕简介、Research Interests、Contact 组织；不填入未经提供的身份、学位或日期。
- [x] 研究介绍只陈述兴趣，不声明已有论文、项目成果或实验能力。
- [x] 采用白底、深灰正文、深蓝强调色，文字层级清楚，头像不过度占据首屏。
- [x] 首页移出站点统计、GitHub 活跃图、随机语录、工具清单与友链等模块。
- [x] 清理学术页面的 Author1 / Venue / PDF # / Coming Soon 等占位和通用 Scholar 首页链接；资源入口有真实资料时再接入。
- [x] 原有笔记全部保持 `state: off`，不修改为公开，不删除原有内容与路由源码；构建阶段排除笔记及附件。
- [x] 桌面与手机可读，导航和联系链接可用，键盘焦点清楚。
- [x] 正式实施完成后运行适当的 Astro 检查、构建和浏览器验证。

## 调研与布局原型

第一方学者与实验室调研已保存于本地 docs/research/；已确定的设计记录位于 docs/design/academic-homepage.md。

布局原型使用本地独立工作树与 prototype/academic-homepage 分支，在现有首页通过 ?variant=A|B|C 对照紧凑阅读、侧栏身份与研究概述、横向编辑式排版。原型不发布笔记、不部署网站，不代表已完成正式改版。

原型的本地快照为 `prototype/academic-homepage` 分支提交 `10596c4`，尚未推送。独立工作目录为 `/home/hrg/workspace/huang2202.github.io-prototype`，运行 `npm run prototype` 后访问 `http://127.0.0.1:4322/?variant=B`。三种结构的桌面及手机截图位于 `docs/design/previews/`。

## 正式实施结果

用户已选定 B。正式主页采用侧栏身份与研究概述，在手机上折为纵向布局；资料、分享卡片及 favicon 使用一致的姓名和正式单位署名。

共享集合查询补齐 `state: off` 过滤；`publication.writing: false` 在加载阶段阻止笔记正文和图片附件进入公开构建。14 篇笔记及原头像未修改，633 个私人附件的产物检查无命中。

生产构建和 Astro 检查通过。浏览器覆盖 1440px、390px、320px，章节导航、键盘操作、旧地址跳转、RSS、搜索及元数据检查通过。验收记录和最终截图见 `docs/design/academic-homepage.md` 与 `docs/design/previews/b-final-*`。本地预览地址为 `http://127.0.0.1:4321/`。

## 后续需求：当前个人简介、具体研究兴趣与机构链接（2026-10-13）

拟用标题：Restore the current academic biography and clarify research interests。

创建状态：本次调用 GitHub Issue 创建接口仍返回 `403: Resource not accessible by integration`，未创建远程 Issue。

用户已确定补充范围：

- 在研究兴趣前补充 About：目前是浙江大学计算机学院一年级直博生，在北京中关村学院联合培养。
- Chunhua Shen 是正式博士导师，作者属于其团队；Hao Chen 提供研究指导，使用准确区分两种关系的英文措辞。
- 使用学校和学院的官方英文名称，不从旧页面推断入学时间，不将联合培养写成联合学位。
- 研究问题一：已学机器人技能对新物体、任务和环境的泛化与适应。
- 研究问题二：结合示范数据与强化学习，高效利用有限的环境交互学习和改进技能。
- 保留 B 版布局与领域标题，研究总述和两个条目的正文合计约 60–100 英文词，以研究兴趣语气表达。
- 为学校和实验室分别添加官网链接，署名仍逐字为 `Zhejiang University, State Key Lab of CAD & CG`。
- 学校链接：`https://www.zju.edu.cn/english/`；实验室链接：`http://www.cad.zju.edu.cn/`。
- 笔记继续保持未公开。

验收：个人简介准确反映当前身份和指导关系，研究文案反映以上问题，机构名称与链接匹配，About / Research / Contact 导航可用，Astro 检查和构建通过，桌面及手机无横向溢出。

实施结果：本地预览已补齐当前个人简介、导师主页、学院与联合培养机构链接及具体研究兴趣。构建和三种屏宽的浏览器验收通过，完整记录位于 `docs/design/academic-homepage.md`。2026-10-13 用户已验收并授权推送部署，发布结果以本轮提交的 GitHub Pages 工作流状态及线上验收为准。
