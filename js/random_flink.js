// 洗牌算法 (Fisher-Yates Shuffle)
function shuffle(a) {
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

// 获取所有友链列表并随机排序
function randomLinkList() {
    // Butterfly 的友链容器类名是 .flink-list
    let linkLists = document.querySelectorAll('.flink-list');
    
    linkLists.forEach(function(list) {
        // 1. 把当前组的所有友链卡片取出来变成一个数组
        let items = Array.from(list.children);
        
        // 2. 如果数量少于2个，就不洗牌了
        if (items.length <= 1) return;
        
        // 3. 洗牌
        shuffle(items);
        
        // 4. 把洗好牌的卡片重新放回去
        items.forEach(function(item) {
            list.appendChild(item);
        });
    });
}

// 页面加载完成后执行
document.addEventListener("DOMContentLoaded", randomLinkList);

// 适配 PJAX (如果你开启了 PJAX，这一步是必须的，否则切换页面后随机失效)
document.addEventListener("pjax:complete", randomLinkList);