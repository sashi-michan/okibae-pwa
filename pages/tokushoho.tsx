import React from 'react'
import { useRouter } from 'next/router'
import Head from 'next/head'

export default function Tokushoho() {
  const router = useRouter()

  return (
    <>
      <Head>
        <title>特定商取引法に基づく表記 - OKIBAE</title>
        <meta name="description" content="OKIBAEの特定商取引法に基づく表記" />
      </Head>
      <div className="main-container">
        <div className="max-w-4xl mx-auto px-8 py-8">
          <div className="mb-6">
            <button
              onClick={() => router.push('/')}
              className="mb-4 text-brand-600 hover:text-brand-700 flex items-center gap-2 transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
              戻る
            </button>
            <div className="flex items-center justify-center gap-3 animate-fade-in mb-2">
              <img
                src="/okibae-icon.svg"
                alt="OKIBAE"
                className="h-8 w-8"
              />
              <h1 className="typography-main-title">OKIBAE</h1>
            </div>
          </div>

          <div className="space-y-8">
            {/* 特定商取引法に基づく表記 */}
            <div className="card animate-slide-up">
              <h2 className="typography-step-title mb-4">
                特定商取引法に基づく表記
              </h2>
              <p className="text-sm text-gray-500 mb-6">最終更新日: 2026年1月7日</p>

              <div className="typography-body text-base leading-relaxed">
                {/* ここに本文を後から挿入 */}
                <table className="w-full border-collapse">
                  <tbody>
                    <tr className="border-b border-pink-100">
                      <th className="text-left py-4 pr-4 font-bold align-top" style={{ color: '#666', minWidth: '180px' }}>
                        販売事業者名
                      </th>
                      <td className="py-4 text-gray-700">
                        おさしみやさん
                      </td>
                    </tr>
                    <tr className="border-b border-pink-100">
                      <th className="text-left py-4 pr-4 font-bold align-top" style={{ color: '#666' }}>
                        運営統括責任者
                      </th>
                      <td className="py-4 text-gray-700">
                        坂崎未知
                      </td>
                    </tr>
                    <tr className="border-b border-pink-100">
                      <th className="text-left py-4 pr-4 font-bold align-top" style={{ color: '#666' }}>
                        所在地
                      </th>
                      <td className="py-4 text-gray-700">
                        〒812-0011 福岡県福岡市博多区博多駅前1丁目23番2号ParkFront博多駅前1丁目5F-B
                      </td>
                    </tr>
                    <tr className="border-b border-pink-100">
                      <th className="text-left py-4 pr-4 font-bold align-top" style={{ color: '#666' }}>
                        電話番号
                      </th>
                      <td className="py-4 text-gray-700">
                        090-6854-1533<br />
                        ※お問い合わせはメールにて承っております。お電話でのサポートは原則行っておりません。
                      </td>
                    </tr>
                    <tr className="border-b border-pink-100">
                      <th className="text-left py-4 pr-4 font-bold align-top" style={{ color: '#666' }}>
                        メールアドレス
                      </th>
                      <td className="py-4 text-gray-700">
                        okibae.help@gmail.com
                      </td>
                    </tr>
                    <tr className="border-b border-pink-100">
                      <th className="text-left py-4 pr-4 font-bold align-top" style={{ color: '#666' }}>
                        販売価格
                      </th>
                      <td className="py-4 text-gray-700">
                        各商品ページに記載<br />
                        ※表示価格は消費税込みです
                      </td>
                    </tr>
                    <tr className="border-b border-pink-100">
                      <th className="text-left py-4 pr-4 font-bold align-top" style={{ color: '#666' }}>
                        商品代金以外の<br />必要料金
                      </th>
                      <td className="py-4 text-gray-700">
                        インターネット接続料金、通信料金<br />
                        ※お客様負担となります
                      </td>
                    </tr>
                    <tr className="border-b border-pink-100">
                      <th className="text-left py-4 pr-4 font-bold align-top" style={{ color: '#666' }}>
                        支払方法
                      </th>
                      <td className="py-4 text-gray-700">
                        クレジットカード決済（Stripe）
                      </td>
                    </tr>
                    <tr className="border-b border-pink-100">
                      <th className="text-left py-4 pr-4 font-bold align-top" style={{ color: '#666' }}>
                        支払時期
                      </th>
                      <td className="py-4 text-gray-700">
                        購入時即時決済
                      </td>
                    </tr>
                    <tr className="border-b border-pink-100">
                      <th className="text-left py-4 pr-4 font-bold align-top" style={{ color: '#666' }}>
                        商品の提供時期
                      </th>
                      <td className="py-4 text-gray-700">
                        決済完了後即時
                      </td>
                    </tr>
                    <tr className="border-b border-pink-100">
                      <th className="text-left py-4 pr-4 font-bold align-top" style={{ color: '#666' }}>
                        返品・キャンセル<br />について
                      </th>
                      <td className="py-4 text-gray-700">
                        デジタルコンテンツの性質上、原則として返品・返金はお受けできません。<br />
                        ただし、システム不具合により利用できなかったと認められる場合に限り対応いたします。
                      </td>
                    </tr>
                    <tr className='border-b border-pink-100'>
                      <th className="text-left py-4 pr-4 font-bold align-top" style={{ color: '#666' }}>
                        動作環境
                      </th>
                      <td className="py-4 text-gray-700">
                        Google Chrome, Safari等の最新ブラウザ
                      </td>
                    </tr>
                  </tbody>
                </table>

                <div className="mt-8 pt-6 border-t border-pink-100">
                  <p className="text-sm text-gray-500">
                    お問い合わせ先: okibae.help@gmail.com
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
