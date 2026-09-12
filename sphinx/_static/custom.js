/* 对齐 ROS 2 文档的两处细节：
   1. 右上角 Edit on GitHub（主题默认被译为中文）
   2. 页脚版权与构建信息（改为 ROS 2 同款英文形式） */
document.addEventListener('DOMContentLoaded', function () {
  // 右上角
  document.querySelectorAll('.wy-breadcrumbs-aside a').forEach(function (a) {
    var t = (a.textContent || '').trim();
    if (t.indexOf('GitHub') !== -1) {
      a.textContent = ' Edit on GitHub';
    }
  });

  // 页脚
  var info = document.querySelector('footer [role="contentinfo"] p');
  if (info) {
    var year = new Date().getFullYear();
    info.innerHTML =
      '&#169; Copyright ' + year + ', 日月星辰.' +
      '<br>Built with <a href="https://www.sphinx-doc.org/">Sphinx</a> using a theme provided by ' +
      '<a href="https://readthedocs.org">Read the Docs</a>.';
  }
});
