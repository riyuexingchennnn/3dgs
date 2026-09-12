/* 对齐 ROS 2 文档的三处细节：
   1. 右上角 Edit on GitHub（主题默认被译为中文）
   2. 页脚版权行（改为英文形式）
   3. 移除主题自带的中文"构建信息"行，避免与上一条重复 */
document.addEventListener('DOMContentLoaded', function () {
  // 1. 右上角
  document.querySelectorAll('.wy-breadcrumbs-aside a').forEach(function (a) {
    if ((a.textContent || '').indexOf('GitHub') !== -1) {
      a.textContent = ' Edit on GitHub';
    }
  });

  // 2. 移除主题自带的中文构建信息行
  document.querySelectorAll('footer p').forEach(function (p) {
    var t = p.textContent || '';
    if (t.indexOf('Sphinx') !== -1 && t.indexOf('利用') !== -1) {
      p.remove();
    }
  });

  // 3. 版权行改为 ROS 2 同款英文形式
  var info = document.querySelector('footer [role="contentinfo"] p');
  if (info) {
    var year = new Date().getFullYear();
    info.innerHTML =
      '&#169; Copyright ' + year + ', 日月星辰.' +
      '<br>Built with <a href="https://www.sphinx-doc.org/">Sphinx</a> using a theme provided by ' +
      '<a href="https://readthedocs.org">Read the Docs</a>.';
  }
});
