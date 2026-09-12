/* 让右上角链接显示为 ROS 2 文档同款的英文文案 */
document.addEventListener('DOMContentLoaded', function () {
  var aside = document.querySelectorAll('.wy-breadcrumbs-aside a');
  aside.forEach(function (a) {
    var t = (a.textContent || '').trim();
    if (t.indexOf('GitHub') !== -1) {
      a.textContent = ' Edit on GitHub';
    }
  });
});
