import { defineCollection, z } from 'astro:content'
import { glob } from 'astro/loaders'

import { theme } from './site.config'
import { publication } from './site.profile'

function removeDupsAndLowerCase(array: string[]) {
  if (!array.length) return array
  const lowercaseItems = array.map((str) => str.toLowerCase())
  const distinctItems = new Set(lowercaseItems)
  return Array.from(distinctItems)
}

function normalizeCategorySlug(value: string) {
  return value.trim().toLowerCase().replace(/\s+/g, '-').replace(/\/+$/g, '')
}

function normalizeCategories(values: string[]) {
  if (!values.length) return values
  const normalized = values.map(normalizeCategorySlug)
  return Array.from(new Set(normalized))
}

const allowedCategorySlugs = new Set((theme.content.categories ?? []).map((c) => c.slug))

// Define blog collection
const blog = defineCollection({
  // Keep private notes and their image imports out of the public build entirely.
  // The route-level state/draft filters still apply when writing is enabled.
  loader: publication.writing
    ? glob({ base: './src/content/blog', pattern: '**/*.{md,mdx}' })
    : async () => [],
  // Required
  schema: ({ image }) =>
    z.object({
      // Required
      title: z.string().max(60),
      description: z.string().max(160),
      publishDate: z.coerce.date(),
      // Optional
      updatedDate: z.coerce.date().optional(),
      heroImage: z
        .object({
          src: z.union([image(), z.string()]),
          alt: z.string().optional(),
          inferSize: z.boolean().optional(),
          width: z.number().optional(),
          height: z.number().optional(),

          color: z.string().optional()
        })
        .optional(),
      categories: z
        .array(z.string())
        .default([])
        .transform(normalizeCategories)
        .refine(
          (values) =>
            allowedCategorySlugs.size === 0 || values.every((v) => allowedCategorySlugs.has(v)),
          { message: 'Invalid category' }
        ),
      tags: z.array(z.string()).default([]).transform(removeDupsAndLowerCase),
      language: z.string().optional(),
      draft: z.boolean().default(false),
      // Special fields
      comment: z.boolean().default(true),
      state: z.enum(['on', 'warning', 'off']).default('on')
    })
})

// Define docs collection
const docs = defineCollection({
  loader: glob({ base: './src/content/docs', pattern: '**/*.{md,mdx}' }),
  schema: () =>
    z.object({
      title: z.string().max(60),
      description: z.string().max(160),
      publishDate: z.coerce.date().optional(),
      updatedDate: z.coerce.date().optional(),
      tags: z.array(z.string()).default([]).transform(removeDupsAndLowerCase),
      draft: z.boolean().default(false),
      // Special fields
      order: z.number().default(999)
    })
})

export const collections = { blog, docs }
