import { mkdir, writeFile } from 'node:fs/promises'
import sharp from 'sharp'

import { profile } from '../src/site.profile.ts'

const escape = (value) =>
  value.replace(
    /[&<>"']/g,
    (character) =>
      ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&apos;'
      })[character]
  )

const card = `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">
  <rect width="1200" height="630" fill="#ffffff"/>
  <path d="M80 84H1120" stroke="#e4e8ee"/>
  <g font-family="Arial, Helvetica, sans-serif">
    <text x="80" y="220" font-size="68" font-weight="600" fill="#283444">${escape(profile.name)}</text>
    <text x="84" y="272" font-size="26" fill="#667587">(${escape(profile.alias)})</text>
    <text x="84" y="375" font-size="32" fill="#173f6b">${escape(profile.interests[0].title)}</text>
    <text x="84" y="423" font-size="28" fill="#173f6b">${escape(profile.interests[1].title)}</text>
    <text x="84" y="512" font-size="22" fill="#4e5c6d">${escape(profile.affiliation)}</text>
    <text x="84" y="564" font-size="20" fill="#667587">huang2202.github.io</text>
  </g>
</svg>`
const icon = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect x="1" y="1" width="62" height="62" rx="12" fill="#ffffff" stroke="#e4e8ee"/>
  <text x="32" y="41" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="26" font-weight="600" fill="#173f6b">GH</text>
</svg>`

await mkdir(new URL('../public/images/', import.meta.url), { recursive: true })
await mkdir(new URL('../public/favicon/', import.meta.url), { recursive: true })
await writeFile(new URL('../public/images/academic-social-card.svg', import.meta.url), card)
await sharp(Buffer.from(card))
  .png()
  .toFile(new URL('../public/images/academic-social-card.png', import.meta.url).pathname)
await writeFile(new URL('../public/favicon/academic.svg', import.meta.url), icon)
console.log('Generated academic social card and favicon from the profile.')
