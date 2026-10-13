# 个人学术主页设计讨论

状态：用户已选择 B（侧栏身份与研究概述），正式首版已完成并通过构建与浏览器验收。2026-10-07 用户验收预览并明确授权推送部署；部署结果以 GitHub Actions 和线上验收为准。参考材料见[网站调研报告](../research/academic-homepage-study.md)，术语见[根目录词汇表](../../GLOSSARY.md)。

## 已确定

- **主要受众**：国际研究同行与潜在研究合作者。学术主页优先帮助他们理解作者研究方向与判断合作匹配度。
- **姓名与别名**：正式学术署名为 `Guangyu Huang`，作者自行选用的公开别名为 `Harry Guang`。姓名区采用 `Guangyu Huang (Harry Guang)`。
- **论文单位署名**：逐字使用 `Zhejiang University, State Key Lab of CAD & CG`。
- **当前角色与培养关系**：作者于 2026-10-13 确认，目前是浙江大学计算机学院一年级直博生，在北京中关村学院联合培养。个人简介采用当前确认的信息，入学年月不从旧页面的矛盾时间线推算。
- **导师与研究指导**：Chunhua Shen 是正式博士导师，作者属于其领导的团队；Hao Chen 提供过很多研究指导。公开表述分别采用 doctoral supervision 和 research guidance，不将后者写成未确认的共同导师关系。
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
- **后续可补充**：有准确时间线、CV 或个人 Scholar 资料时再接入相应内容。已有公开成果后再扩展研究工作条目。当前身份、博士导师与联合培养信息已于 2026-10-13 补充确认。
- **本阶段不展示**：论文占位、项目占位、站点经营统计、GitHub 活跃图、随机语录、工具清单、友链和稿件列表。

现有博客内容、未公开状态和地址定义继续保留。未来公开任何笔记时另行确定公开范围。用户已验收本地改版并授权推送与部署；现有笔记仍保持未公开。

## 姓名区内容

2026-10-07 首版的姓名区省略了当时尚未明确的个人角色字段；2026-10-13 更新的身份介绍见后文：

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

以上英文陈述用于原型和首版文案。后续已确定的研究问题与个人简介见 2026-10-13 更新。

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

## 小幅补充讨论（2026-10-13）

状态：用户已接受补充具体研究问题与机构官网链接，并选定“泛化与适应”和“数据与交互效率”两条主线；随后确认当前个人身份与指导关系，本轮一并补充 About。2026-10-13 用户验收预览并明确授权部署。

原研究简介和两个兴趣条目都在描述宽泛领域，读者尚不能判断作者具体关心哪些研究问题。最初只检索了改版后的文件，未找到个人 Bio；用户提醒后从 Git 历史找回了学院、指导老师和身份介绍，见下文。未发现可直接使用的个人 CV。

已确定的增量：把现有研究介绍写得更具体，用约 60–100 个英文词说明 1–2 个真实关注的问题，以及强化学习与这些问题的关系。整合现有段落与条目，保持 B 版的紧凑程度。这里的内容仍是研究兴趣；只有作者确认实际在做的工作，才采用进行中的研究工作表述。

同时为学校与实验室名称增加经过核实的官网链接，保留已确定的机构署名原文。机构链接帮助读者了解机构，不用作作者个人身份的核验证据。作者随后补充了当前身份与指导关系，本轮一并恢复简短个人介绍；CV 仍待准确资料。

本轮参考 [Chelsea Finn 的主页](https://ai.stanford.edu/~cbfinn/) 与 [Sergey Levine 的主页](https://people.eecs.berkeley.edu/~svlevine/)：二者都在开头用简短文字交代身份和研究目标。借鉴其信息组织方式，具体研究目标由作者确定。

用户已选择的两个问题：

- 泛化与适应：机器人如何将学到的技能应用到新的物体、任务与环境，并在条件变化时调整行为。
- 数据与交互效率：如何结合示范数据与强化学习，通过有限的环境交互学习和改进技能。

保留现有领域标题与简短总述，把两个兴趣条目的正文写具体；总述与两段正文合计约 70 个英文词。具体文案：

> My interests center on how robots can generalize learned skills to new objects, tasks, and environments, and adapt their behavior when conditions change.

> I am interested in combining demonstration data with reinforcement learning to acquire and refine skills, with an emphasis on making efficient use of limited interaction with the environment.

核实后的机构链接为 [浙江大学英文官网](https://www.zju.edu.cn/english/) 与 [CAD&CG 实验室中文官网](http://www.cad.zju.edu.cn/)。实验室的 HTTPS 握手失败，`/english/` 返回 404，采用可访问的 HTTP 中文主页。学校和实验室分别链接，展示文本连起来仍逐字为 `Zhejiang University, State Key Lab of CAD & CG`。

### 从旧页面找回的个人介绍

用户提醒原页面包含导师与学院信息后，追溯了 B 版改造前提交 `65338fea17ef556f046bb758c4957871326dd861`。

- 旧 `src/pages/academic/index.astro` 写的是西安交通大学人工智能学院 / 钱学森荣誉学院的本科四年级学生；当时与 [Hao Chen](https://stan-haochen.github.io/) 合作，并计划在 [Chunhua Shen](https://cshen.github.io/) 指导下加入 ZJU CAD&CG。
- 旧学院链接为 `http://www.aiar.xjtu.edu.cn/` 和 `https://bjb.xjtu.edu.cn/`；旧 `ZJU CAD&CG` 链接误指向 `https://cyber.zju.edu.cn`，恢复简介时不复用该链接。
- 旧首页中文 About 写“目前大四在浙江大学提前进组工作中”；Education 卡则写“直博生 / Sep 2025 - Present”和西安交大本科“Sep 2022 - Sep 2026”。这些表述不能同时作为当前身份事实直接恢复。

作者随后确认了现状：浙江大学计算机学院一年级直博生，在北京中关村学院联合培养；正式博士导师为 Chunhua Shen，属于其领导的团队；Hao Chen 给予了很多研究指导。这些当前信息取代旧页中关于未来入组、争取机会和本科四年级的陈述。个人简介不额外推断入学时间、联合学位或共同导师关系。

### 当前个人简介

在 Research 前增加简短 About，导航顺序为 About / Research / Contact。正文与链接仍统一保存在 `src/site.profile.ts`：

> I am a first-year Ph.D. student in the direct-entry program at the College of Computer Science and Technology, Zhejiang University. I am jointly trained at Zhongguancun Academy in Beijing.

> I am advised by Prof. Chunhua Shen and am a member of his group at the State Key Lab of CAD & CG. I have also benefited greatly from Hao Chen’s research guidance.

机构英文名和链接于 2026-10-13 核实：

- [浙江大学官方 Schools & Colleges 目录](https://www.zju.edu.cn/english/74913/list.psp) 使用 `College of Computer Science and Technology`，页面 HTTPS 可访问。目录指向的学院英文子站当前超时，因此简介暂链接官方校级目录；同名[学院介绍](https://www.zju.edu.cn/english/2018/0523/c20093a813217/page.htm)可访问，但日期为 2018 年。
- [中关村学院英文官网](https://www.bza.edu.cn/en/)和 [About 页](https://www.bza.edu.cn/en/about)正文使用 `Zhongguancun Academy`。新域名的 HTTPS 可访问；旧 `bjzgca.edu.cn` HTTPS 证书过期，不再使用。
- 两位老师分别链接 [Chunhua Shen 个人主页](https://cshen.github.io/)和 [Hao Chen 个人主页](https://stan-haochen.github.io/)。培养和指导关系来自作者确认，机构页面仅核对名称与链接。

本轮验收完成：研究总述与两段正文合计 67 个英文词；Astro 检查与生产构建通过（0 errors、0 warnings，保留原有 Giscus 提示）。1440px、390px、320px 均无横向溢出，简介的完整文本和四个链接正确，About / Research / Contact 导航与当前章节标记可用；页面和结构化数据中的机构署名完全一致，RSS 仍为 0 条。笔记源码、可见性配置及头像未修改。

截图见 [桌面](./previews/research-focus-desktop.png) 与 [手机](./previews/research-focus-mobile.png)，浏览器记录见 [checks](./previews/research-focus-checks.json)。本地预览为 `http://127.0.0.1:4321/`。用户已验收并授权将本轮改动推送至 `main`，由现有 [GitHub Pages 工作流](https://github.com/huang2202/huang2202.github.io/actions/workflows/deploy.yml)发布；部署结果以该提交的工作流状态及线上验收为准。
