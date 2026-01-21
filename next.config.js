const withPWA = require('next-pwa')({
  dest: 'public',
  register: true,
  skipWaiting: true,
  disable: true, // 一時的に完全無効化してログイン問題を診断
});
module.exports = withPWA({ reactStrictMode: true });
