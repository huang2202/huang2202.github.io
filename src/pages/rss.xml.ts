import type { APIRoute } from 'astro'
import rss from '@astrojs/rss'
import config from 'virtual:config'

import { getBlogCollection, sortMDByDate } from 'astro-pure/server'

export const GET: APIRoute = async ({ site }) => {
  const posts = sortMDByDate<'blog'>(await getBlogCollection('blog'))
  return rss({
    title: config.title,
    description: config.description,
    site: site ?? import.meta.env.SITE,
    trailingSlash: false,
    stylesheet: '/scripts/pretty-feed-v3.xsl',
    items: posts.map(({ id, data }) => ({
      title: data.title,
      description: data.description,
      pubDate: data.publishDate,
      link: `/blog/${id}`,
      categories: data.categories
    }))
  })
}
