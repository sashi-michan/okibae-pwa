// PWA一時無効化 - デバッグ用
// const withPWA = require('next-pwa')({
//   dest: 'public',
//   register: true,
//   skipWaiting: true,
//   disable: process.env.NODE_ENV === 'development',
// });
// module.exports = withPWA({ reactStrictMode: true });

module.exports = {
  reactStrictMode: true
};
