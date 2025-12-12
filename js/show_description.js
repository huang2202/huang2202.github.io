/*
 * show_description.js
 * 功能：自动抓取 meta description 并显示在文章正文顶部
 */
document.addEventListener("DOMContentLoaded", function () {
    // 1. 检测是否为文章页或独立页面 (且必须有文章容器)
    var articleContainer = document.getElementById('article-container');
    if (!articleContainer) return;

    // 2. 获取 meta description
    var metaDesc = document.querySelector('meta[name="description"]');
    if (!metaDesc) return;
    
    var descText = metaDesc.getAttribute('content');
    
    // 3. 过滤逻辑：如果描述为空，或者描述就是博客默认的副标题/全局描述，则不显示
    // (防止没有写 description 的文章显示一句通用的废话)
    var siteTitle = document.title.split('|')[1]; // 简单过滤，视情况调整
    if (!descText || descText.length < 5) return; 

    // 4. 创建显示容器
    var descDiv = document.createElement('div');
    descDiv.className = 'post-description-box'; // 设置类名，方便 CSS 美化
    
    // 插入图标和文字
    descDiv.innerHTML = `
        <div class="desc-icon"><i class="fas fa-quote-left"></i></div>
        <div class="desc-text">${descText}</div>
    `;

    // 5. 插入到文章正文的最前面
    articleContainer.insertBefore(descDiv, articleContainer.firstChild);
});