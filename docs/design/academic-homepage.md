# 个人学术主页设计讨论

状态：用户已选择 B（侧栏身份与研究概述），正式首版已完成并通过构建与浏览器验收。2026-10-07 用户验收预览并明确授权推送部署；部署结果以 GitHub Actions 和线上验收为准。参考材料见[网站调研报告](../research/academic-homepage-study.md)，术语见[根目录词汇表](../../GLOSSARY.md)。

## 已确定

- **主要受众**：国际研究同行与潜在研究合作者。学术主页优先帮助他们理解作者研究方向与判断合作匹配度。
- **姓名与别名**：正式学术署名为 `Guangyu Huang`，作者自行选用的公开别名为 `Harry Guang`。姓名区采用 `Guangyu Huang (Harry Guang)`。
- **论文单位署名**：逐字使用 `Zhejiang University, State Key Lab of CAD & CG`。
- **当前角色与年份**：本轮没有提供具体资料，首版采用只列机构的方式，省略学位阶段、个人角色和开始年份；后续有准确资料时再补充。
- **头像**：暂时沿用现有动漫头像。用户目前没有合适的个人照片，改版不依赖提供真人照片。
- **Writing**：同站保留研究阅读、技术笔记和个人写作，学术主页突出研究内容。
- **内容与路径**：保留现有笔记内容和路由源码；现阶段任何笔记均不公开，保留全部笔记原有的 `state: off`。学术首页和导航暂不展示 Writing 入口或精选笔记。
- **语言策略**：主页与研究介绍用英文，Writing 保持文章原语言；精选中文文章可以补充英文摘要。
- **公开成果现状**：用户暂时没有公开研究成果，第一版先建立研究兴趣内容。暂不规划依赖公开论文或项目素材的成果展示。
- **研究兴趣**：以 `Robot Learning & Embodied Intelligence` 为主线，将 `Reinforcement Learning` 作为方法方向。具体研究成果与实验能力不由兴趣表述推定。
- **视觉取向**：现代、克制、分区清楚的学术页面，采用白底、深灰正文、深蓝强调色和小尺寸动漫头像。

## 首版内容规格

- **页眉**：使用正式学术姓名；导航以 Profile、Research、Contact 等有实际内容的章节为主。
- **简介**：正式姓名、公开别名、小尺寸现有头像、机构名称与简短英文研究兴趣陈述。
- **Research Interests**：Robot Learning & Embodied Intelligence，以及作为方法方向的 Reinforcement Learning。
- **Contact**：沿用源码中的邮箱 `zju_hgy@163.com` 和 GitHub `https://github.com/huang2202`。
- **后续可补充**：有准确角色、时间线、CV 或个人 Scholar 资料时再接入相应内容。已有公开成果后再扩展研究工作条目。
- **本阶段不展示**：论文占位、项目占位、站点经营统计、GitHub 活跃图、随机语录、工具清单、友链和稿件列表。

现有博客内容、未公开状态和地址定义继续保留。未来公开任何笔记时另行确定公开范围。用户已验收本地改版并授权推送与部署；现有笔记仍保持未公开。

## 姓名区内容

首版省略尚未给出具体信息的个人角色字段：

```text
Guangyu Huang (Harry Guang)
Zhejiang University, State Key Lab of CAD & CG
```

头像采用适度尺寸，周围保留清晰的文字层级。具体布局通过独立原型对照；原型只含已经明确的资料。

## 英文文案草稿

下面的兴趣陈述只表达已确定的研究兴趣，不声明已有研究成果或公开项目：

> My research interests lie in robot learning and embodied intelligence, with a focus on reinforcement learning.

Research Interests 可按两个方向组织：

- **Robot Learning & Embodied Intelligence** — I am interested in how embodied agents acquire intelligent behavior through learning and interaction.
- **Reinforcement Learning** — I am interested in learning methods for sequential decision-making and control.

个人角色和经历时间线后续有准确资料时补充；以上英文陈述用于原型和首版文案。

## 原型问题

在暂无公开成果、笔记保持草稿、动漫头像保留的条件下，怎样安排身份、研究兴趣与联系入口，才能形成克制、清晰的个人学术主页？

原型在独立工作树及 `prototype/academic-homepage` 分支制作。现有首页路由通过 `?variant=A|B|C` 对照三种结构：紧凑阅读、侧栏身份与研究概述、横向编辑式排版。三者均保持已确定的配色、资料和公开范围。

## 原型预览与验证

独立工作目录：`/home/hrg/workspace/huang2202.github.io-prototype`。在该目录运行 `npm run prototype`，即可访问 `http://127.0.0.1:4322/`。默认展示 B，底部按钮和左右方向键可以切换，URL 中的 variant 参数可分享并在刷新后保留。

原型已捕获为本地分支 `prototype/academic-homepage` 的提交 `10596c4`，未推送到远程。该提交保留三种布局、切换器、已确定的设计资料和视觉证据。

| 版本 | 结构                           | 桌面预览                         | 手机预览                        |
| ---- | ------------------------------ | -------------------------------- | ------------------------------- |
| A    | 紧凑学术阅读                   | [桌面](./previews/a-desktop.png) | [手机](./previews/a-mobile.png) |
| B    | 侧栏身份与研究概述；用户已选定 | [桌面](./previews/b-desktop.png) | [手机](./previews/b-mobile.png) |
| C    | 横向编辑式个人页               | [桌面](./previews/c-desktop.png) | [手机](./previews/c-mobile.png) |

浏览器检查覆盖三个版本的 1440px 桌面和 390px 手机视口，均未出现横向溢出；按钮、方向键、URL 切换和无效 variant 的默认回退正常。原型没有新增测试套件，也没有执行正式生产构建。

GitHub Issue 创建受到集成写入权限限制，返回 `403: Resource not accessible by integration`。完整需求正文已保存为 [Issue 草稿](./academic-homepage-issue-draft.md)，尚未在 GitHub 创建。

## B 版正式实施与验收

正式主页由 `src/components/home/AcademicProfile.astro` 与 `src/layouts/AcademicLayout.astro` 实现。桌面采用身份侧栏和研究概述，手机改为纵向布局。动漫头像显示为桌面 86px、手机 70px；研究兴趣、联系方式和正式单位署名均使用已确定的内容，不补造身份或成果。

`src/site.profile.ts` 集中保存身份、研究兴趣、联系方式和写作公开开关，首页、全站元数据、分享卡片与 favicon 使用一致的资料。正式页面不包含原型切换器。原型的 A/B/C 仍保存在独立工作树和本地分支中。

旧地址 `/academic`、`/projects` 跳转到 `/#research`，`/about` 跳转到 `/#profile`，`/links` 跳转到 `/#contact`。`/blog` 保留公开写作索引，目前为空。仓库沿用 `trailingSlash: never`，本地预览检查使用无尾斜杠地址。

原主题的集合查询只过滤生产草稿，未统一过滤 `state: off`，可能把未公开笔记加入 RSS、归档或标签数据；此处已在共享查询入口补齐过滤。另发现仅隐藏文章路由仍会让 Markdown 图片进入静态产物，首版因此将 `publication.writing` 设为 `false`，在内容加载阶段排除全部笔记和附件。14 篇笔记及头像原文件没有修改。

开发与 CI 使用 Node.js 22，构建自动生成分享图和姓名缩写 favicon，依赖版本未升级。

| 验收项                                         | 结果                                                                     |
| ---------------------------------------------- | ------------------------------------------------------------------------ |
| Astro 类型检查与生产构建                       | 通过；0 errors、0 warnings，保留原有 Giscus script 提示                  |
| 1440px 桌面、390px 与 320px 手机               | 无横向溢出；身份、头像和正文布局正常                                     |
| 章节导航、键盘焦点与 Skip to content           | 通过                                                                     |
| 旧地址跳转、邮箱与 GitHub 地址                 | 通过                                                                     |
| 姓名、机构、canonical、分享图与 Person JSON-LD | 与确定资料一致                                                           |
| 笔记公开范围                                   | RSS 0 条；私人文章返回 404；搜索不含私人笔记                             |
| 私人附件与正文产物检查                         | 633 个附件与 56 个公开文本产物检查通过，无私人图片、文件名或笔记标题命中 |
| 源码与格式                                     | 笔记及原头像 diff 为空；变更文件已格式化，`git diff --check` 通过        |

正式本地预览：`http://127.0.0.1:4321/`（`npm run preview`）。桌面截图见 [B 正式版](./previews/b-final-desktop.png)，手机截图见 [B 正式版手机](./previews/b-final-mobile.png)。浏览器验收数据见 [checks](./previews/b-final-checks.json)，发布范围检查见 [publication checks](./previews/b-final-publication-checks.json)。
