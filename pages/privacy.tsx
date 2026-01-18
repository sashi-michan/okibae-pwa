import React from 'react'
import { useRouter } from 'next/router'
import Head from 'next/head'

export default function Privacy() {
  const router = useRouter()

  return (
    <>
      <Head>
        <title>プライバシーポリシー - OKIBAE</title>
        <meta name="description" content="OKIBAEのプライバシーポリシー" />
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
            {/* プライバシーポリシー */}
            <div className="card animate-slide-up">
              <h2 className="typography-step-title mb-4">
                プライバシーポリシー
              </h2>
              <p className="text-sm text-gray-500 mb-6">最終更新日: 2026年1月7日</p>

              <div className="space-y-6 typography-body text-base leading-relaxed">
                {/* ここに本文を後から挿入 */}
                <section>
                  <p className="text-gray-700">
                    おさしみやさん（以下「運営者」）は、OKIBAE（以下「本サービス」）におけるユーザーの情報の取り扱いについて、以下のとおりプライバシーポリシー（以下「本ポリシー」）を定めます。
                  </p>
                </section>
                <section>
                  <h3 className="text-lg font-bold mb-3" style={{ color: '#666' }}>
                    1. 取得する情報
                  </h3>
                  <p className="text-gray-700 mb-3">
                    運営者は、本サービスの提供にあたり、以下の情報を取得する場合があります：
                  </p>
                  <ul className="list-disc list-inside text-gray-700 ml-4 space-y-1">
                    <li>アカウント登録情報（メールアドレス等）</li>
                    <li>決済に関する情報（決済手段に関する情報の一部、購入履歴等）</li>
                    <li>端末情報、ログ情報、Cookie等（アクセス解析、障害対応、不正対策のため）</li>
                    <li>お問い合わせ内容</li>
                  </ul>
                  <p className="text-gray-700 mt-3">
                    ※クレジットカード番号等の決済情報は、決済事業者（Stripe）により取り扱われ、運営者が保持しません。
                  </p>
                </section>

                <section>
                  <h3 className="text-lg font-bold mb-3" style={{ color: '#666' }}>
                    2. 利用目的
                  </h3>
                  <p className="text-gray-700 mb-3">
                    取得した情報は、以下の目的で利用します：
                  </p>
                  <ul className="list-disc list-inside text-gray-700 ml-4 space-y-1">
                    <li>本サービスの提供・運営・改善</li>
                    <li>本人確認、認証、アカウント管理</li>
                    <li>決済処理、購入履歴の管理</li>
                    <li>不正利用防止、セキュリティ確保</li>
                    <li>お問い合わせ対応</li>
                  </ul>
                </section>

                <section>
                  <h3 className="text-lg font-bold mb-3" style={{ color: '#666' }}>
                    3. 外部サービスの利用
                  </h3>
                  <p className="text-gray-700 mb-3">
                    本サービスでは、以下の外部サービスを利用する場合があります：
                  </p>
                  <ul className="list-disc list-inside text-gray-700 ml-4 space-y-1">
                    <li>生成AI関連サービス（例：Google Vertex AI）：画像生成等の処理</li>
                    <li>決済事業者（Stripe）：クレジットカード決済の処理</li>
                    <li>ホスティング等（Vercel）：アプリケーションのデプロイ</li>
                    <li>その他、運営者が定める外部サービス</li>
                  </ul>
                  <p className="text-gray-700 mt-3">
                    外部サービスに送信される情報や当該サービス側での情報の取り扱いは、各サービスの規約・プライバシーポリシーに従います。<br />
                    決済に必要な情報（メールアドレス等）が決済事業者（Stripe）に送信・保存される場合があります。
                  </p>
                </section>

                <section>
                  <h3 className="text-lg font-bold mb-3" style={{ color: '#666' }}>
                    4. 入力データおよび生成結果（画像）の取り扱い
                  </h3>
                  <p className="text-gray-700">
                    運営者は、入力データおよび生成結果（画像）を本サービス上に保存しません。<br />
                    ただし、画像生成等の処理のため、外部サービスへ送信される場合があります。
                  </p>
                </section>

                <section>
                  <h3 className="text-lg font-bold mb-3" style={{ color: '#666' }}>
                    5. 退会後の情報の取り扱い
                  </h3>
                  <p className="text-gray-700">
                    退会後、運営者はアカウントを利用できない状態にし、運営上不要になり次第、合理的な範囲で削除または匿名化を行います。<br />
                    ただし、法令対応、会計・税務上の記録保持、不正利用防止のために必要な情報（購入履歴等）は、退会後も一定期間保持する場合があります。<br />
                    また、無料クレジットの重複付与防止等のため、メールアドレス等を元に作成した復元できない形式(ハッシュ化等)の識別子を保持する場合があります。
                  </p>
                </section>

                <section>
                  <h3 className="text-lg font-bold mb-3" style={{ color: '#666' }}>
                    6. 安全管理措置
                  </h3>
                  <p className="text-gray-700">
                    運営者は、取得した情報の漏えい、滅失、毀損等を防止するため、合理的な安全管理措置を講じます。
                  </p>
                </section>

                <section>
                  <h3 className="text-lg font-bold mb-3" style={{ color: '#666' }}>
                    7. 開示・訂正・削除等の請求
                  </h3>
                  <p className="text-gray-700">
                    ユーザーは、運営者が保有する自己の情報について、法令に基づき開示・訂正・削除等を求めることができます。お問い合わせ窓口よりご連絡ください。
                  </p>
                </section>

                <section>
                  <h3 className="text-lg font-bold mb-3" style={{ color: '#666' }}>
                    8. お問い合わせ窓口
                  </h3>
                  <p className="text-gray-700">
                  【問い合わせ窓口メールアドレス】<br />
                  okibae.help@gmail.com
                  </p>
                </section>

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
