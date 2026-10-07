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
- 个人角色和开始年份未提供，首版只列机构。
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
