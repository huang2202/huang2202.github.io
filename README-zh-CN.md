# Guangyu Huang — 个人学术主页

Guangyu Huang（Harry Guang）的个人学术网站，研究兴趣为机器人学习、具身智能与强化学习。

正式单位署名：**Zhejiang University, State Key Lab of CAD & CG**。

首页采用已选定的 B 版布局：左侧身份与联系入口，右侧研究概述。原有笔记全部保持未公开。

## 本地开发

使用 Node.js 22（22.18.0 或更高的 22.x 版本），版本要求见 `.nvmrc` 和部署工作流。

```sh
npm ci --legacy-peer-deps
npm run dev
npm run build
npm run preview
```

构建会生成个人分享卡片与 favicon，执行 Astro 类型检查、静态构建及搜索索引生成。单独更新图像资源可运行 `npm run assets:profile`。

## 内容与公开范围

- `src/site.profile.ts`：姓名、别名、单位、研究兴趣、联系方式与写作公开开关。
- `src/components/home/AcademicProfile.astro`：公开学术主页。
- `src/site.config.ts`：全站信息与导航。
- `src/content/blog/`：原有研究和个人笔记。
- `docs/design/academic-homepage.md`：已确定的设计与公开范围。
- `docs/research/`：学者和实验室官网调研。

当前 `publication.writing` 为 `false`，构建不加载任何笔记，正文和图片附件均不会进入公开目录。将来启用写作后，`state: off` 的笔记仍会从文章页、RSS、归档、标签与分类数据中排除，生产构建也会排除草稿。公开笔记需依据作者明确的公开范围调整，本次改版保留全部笔记文件及原有状态。

旧的 `/academic`、`/about`、`/projects`、`/links` 地址保留，并跳转到对应首页章节。`/blog` 继续作为公开写作索引，目前显示无已发布文章的状态。

## 技术基础

使用 [Astro](https://astro.build/) 和 [Astro Pure](https://github.com/cworld1/astro-theme-pure)。保留上游 [LICENSE](./LICENSE)。
