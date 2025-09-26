import type { AppProps } from 'next/app'
import Head from 'next/head'
import '../styles/globals.css'
import NavBar from '../components/NavBar'
export default function MyApp({ Component, pageProps }: AppProps) {
  return (
    <>
      <Head>
        <title>OKIBAE - おしゃれな置き画を、かんたんに</title>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
        <meta name="theme-color" content="#C2A2A8" />

        {/* ファビコン設定 */}
        <link rel="icon" href="/okibae-icon.svg" type="image/svg+xml" />
        <link rel="icon" href="/icons/icon-192.png" type="image/png" sizes="192x192" />
        <link rel="apple-touch-icon" href="/icons/icon-192.png" />

        {/* PWA設定 */}
        <link rel="manifest" href="/manifest.json" />
        <meta name="application-name" content="OKIBAE" />
        <meta name="description" content="手作り作家さん向けの商品撮影背景置き換えアプリ。AIで美しい置き画を簡単に作成" />
      </Head>
      <div className="min-h-full flex flex-col">
        <NavBar />
        <main className="flex-1 mx-auto w-full max-w-3xl px-4 py-6 md:py-10">
          <Component {...pageProps} />
        </main>
        <footer className="text-center text-xs text-gray-500 py-6">
          <span>© {new Date().getFullYear()} OKIBAE</span>
        </footer>
      </div>
    </>
  )
}
