# Guangyu Huang — Academic homepage

Personal academic website for Guangyu Huang (Harry Guang), focused on robot learning, embodied intelligence, and reinforcement learning.

Affiliation: **Zhejiang University, State Key Lab of CAD & CG**.

The homepage uses the selected B layout: a compact identity sidebar and a research overview. Existing notes remain unpublished.

## Development

Use Node.js 22 (22.18.0 or later), as specified in `.nvmrc` and the deployment workflow.

```sh
npm ci --legacy-peer-deps
npm run dev
npm run build
npm run preview
```

The build generates the profile's social card and favicon, checks Astro types, builds static pages, and creates the search index. To regenerate profile assets separately, run `npm run assets:profile`.

## Content

- `src/site.profile.ts`: name, alias, affiliation, research interests, contact links, and writing publication setting.
- `src/components/home/AcademicProfile.astro`: the public academic profile.
- `src/site.config.ts`: shared site metadata and navigation.
- `src/content/blog/`: existing research and personal notes.
- `docs/design/academic-homepage.md`: agreed design and publication scope.
- `docs/research/`: scholar and laboratory website references.

`publication.writing` is currently `false`: notes are not loaded into the public build, so their bodies and image attachments stay private. When writing is enabled, entries marked `state: off` are still excluded from article routes, RSS, archives, and tag/category data; drafts are also excluded from production. Publishing notes requires an explicit change to the author's agreed publication scope. The redesign leaves all existing note files and flags intact.

Legacy `/academic`, `/about`, `/projects`, and `/links` addresses lead to the relevant homepage sections. `/blog` remains a published-writing index with an empty state while no notes are public.

## Foundation

Built with [Astro](https://astro.build/) and [Astro Pure](https://github.com/cworld1/astro-theme-pure). The upstream license is retained in [LICENSE](./LICENSE).
