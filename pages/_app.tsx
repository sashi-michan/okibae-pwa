import type { AppProps } from 'next/app'
import Head from 'next/head'
import '../styles/globals.css'
import NavBar from '../components/NavBar'
export default function MyApp({ Component, pageProps }: AppProps) {
  return (
    <>
      <Head>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
        <meta name="theme-color" content="#e27f7f" />
        <link rel="manifest" href="/manifest.json" />
        <link rel="apple-touch-icon" href="/icons/icon-192.png" />
        <meta name="application-name" content="OKIBAE" />
        <meta name="description" content="おしゃれな置き画作成アプリ（PWA）" />
      </Head>
      <div className="min-h-full flex flex-col">
        <NavBar />
        <main className="flex-1 mx-auto w-full max-w-3xl px-4 py-6 md:py-10">
          <Component {...pageProps} />
        </main>
        <footer className="text-center text-xs text-gray-500 py-6">
          <span>© {new Date().getFullYear()} OKIBAE Starter</span>
        </footer>
      </div>
    </>
  )
}
