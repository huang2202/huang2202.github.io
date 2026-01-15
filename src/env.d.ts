/// <reference types="astro/client" />

declare module 'virtual:astro:assets/fonts/internal' {
  export const internalConsumableMap:
    | Map<unknown, { css: string; preloadData: any[] }>
    | undefined
}
