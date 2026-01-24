import React from 'react'
import { useRouter } from 'next/router'
import Head from 'next/head'

export default function Terms() {
  const router = useRouter()

  return (
    <>
      <Head>
        <title>利用規約 - OKIBAE</title>
        <meta name="description" content="OKIBAEの利用規約" />
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
            {/* 利用規約 */}
            <div className="card animate-slide-up">
              <h2 className="typography-step-title mb-4">
                利用規約
              </h2>
              <p className="text-sm text-gray-500 mb-6">最終更新日: 2026年1月7日</p>

              <div className="space-y-6 typography-body text-base leading-relaxed">
                {/* ここに本文を後から挿入 */}
                <section>
                  <p className="text-gray-700">
                    この利用規約（以下「本規約」）は、おさしみやさん（以下「運営者」）が提供する「OKIBAE」（以下「本サービス」）の利用条件を定めるものです。<br />
                    本サービスを利用することにより、本規約に同意したものとみなされます。
                  </p>
                </section>
                <section>
                  <h3 className="text-lg font-bold mb-3" style={{ color: '#666' }}>
                    第1条（適用範囲）
                  </h3>
                  <ol className="list-decimal list-inside text-gray-700 space-y-2 ml-4">
                    <li>本規約は、本サービスの利用に関する運営者とユーザーとの間の一切の関係に適用されます。</li>
                    <li>運営者が本サービス上で掲載するルール、ガイド、注意事項等は、本規約の一部を構成します。</li>
                  </ol>
                </section>

                <section>
                  <h3 className="text-lg font-bold mb-3" style={{ color: '#666' }}>
                    第2条（定義）
                  </h3>
                  <p className="text-gray-700 mb-3">
                    本規約において使用する用語は、以下のとおり定義します。
                  </p>
                  <ul className="list-decimal list-inside text-gray-700 space-y-2 ml-4">
                    <li>「ユーザー」：本規約に同意し、本サービスを利用する方</li>
                    <li>「入力データ」：ユーザーが本サービスにアップロードする画像等の情報</li>
                    <li>「生成結果」：本サービスにより生成された画像等の出力</li>
                  </ul>
                </section>

                <section>
                  <h3 className="text-lg font-bold mb-3" style={{ color: '#666' }}>
                    第3条（利用登録）
                  </h3>
                  <ol className="list-decimal list-inside text-gray-700 space-y-2 ml-4">
                    <li>本サービスの利用にあたり、ユーザーは運営者が定める方法により利用登録を行う必要があります。</li>
                    <li>運営者は、登録申請に虚偽がある場合、その他運営者が不適切と判断した場合、登録を承認しないことがあります。</li>
                  </ol>
                </section>

                <section>
                  <h3 className="text-lg font-bold mb-3" style={{ color: '#666' }}>
                    第4条（アカウント管理）
                  </h3>
                  <ol className="list-decimal list-inside text-gray-700 space-y-2 ml-4">
                    <li>ユーザーは、自己の責任においてアカウント情報を管理するものとします。</li>
                    <li>アカウントの不正利用によりユーザーに損害が生じた場合でも、運営者は運営者の故意または重大な過失がない限り責任を負いません。</li>
                  </ol>
                </section>

                <section>
                  <h3 className="text-lg font-bold mb-3" style={{ color: '#666' }}>
                    第5条（クレジット）
                  </h3>
                  <ol className="list-decimal list-inside text-gray-700 space-y-2 ml-4">
                    <li>本サービスでは、画像生成等の機能利用にあたりクレジットを消費します。</li>
                    <li>1クレジットで1回の生成が可能です。</li>
                    <li>クレジットの有効期限はありません。</li>
                    <li>クレジットは、本サービスの利用にあたり、運営者が定める方法により購入することができます。</li>
                    <li>ユーザー都合によるクレジットの返金・払い戻しは、法令で認められる場合を除き行いません。</li>
                    <li>無料クレジット（初回特典等）を提供する場合、原則としてお一人につき初回登録時の1回のみ付与します。同一の方が再登録した場合など、運営者が同一ユーザーと判断できる場合には、無料クレジットを付与しないことがあります。</li>
                  </ol>
                </section>

                <section>
                  <h3 className="text-lg font-bold mb-3" style={{ color: '#666' }}>
                    第6条（禁止事項）
                  </h3>
                  <p className="text-gray-700 mb-3">
                  ユーザーは、本サービスの利用にあたり、以下の行為をしてはなりません。
                  </p>
                  <ol className="list-decimal list-inside text-gray-700 space-y-2 ml-4">
                    <li>法令または公序良俗に反する行為</li>
                    <li>犯罪行為に関連する行為</li>
                    <li>運営者または第三者の知的財産権（著作権、商標権等）、肖像権、プライバシーその他の権利・利益を侵害する行為</li>
                    <li>不正アクセス、過度な負荷をかける行為、または本サービスの運営を妨げる行為</li>
                    <li>本サービスの不具合を意図的に利用する行為、または無料クレジット等の特典を不正に取得しようとする行為</li>
                    <li>反社会的勢力に対する利益供与その他これに準ずる行為</li>
                    <li>その他、運営者が不適切と判断する行為</li>
                  </ol>
                </section>

                <section>
                  <h3 className="text-lg font-bold mb-3" style={{ color: '#666' }}>
                    第7条（本サービスの提供・変更・停止）
                  </h3>
                  <ol className="list-decimal list-inside text-gray-700 space-y-2 ml-4">
                    <li>運営者は、ユーザーへの事前の通知なく、本サービスの内容を変更し、または提供を中断・停止することがあります。</li>
                    <li>運営者は、前項によりユーザーに生じた損害について、運営者の故意または重大な過失がない限り責任を負いません。</li>
                  </ol>
                </section>

                <section>
                  <h3 className="text-lg font-bold mb-3" style={{ color: '#666' }}>
                    第8条（生成結果の取り扱い）
                  </h3>
                  <ol className="list-decimal list-inside text-gray-700 space-y-2 ml-4">
                    <li>本サービスは外部の生成AIサービス等を利用しており、生成結果は入力データや外部サービスの仕様・状態等により変動します。</li>
                    <li>運営者は、生成結果の正確性、完全性、特定目的への適合性を保証しません。</li>
                    <li>生成結果は、私的利用に加えて、商業目的（例：ECサイト、フリマアプリ、SNS投稿、広告素材としての利用等）にもご利用いただけます。</li>
                    <li>ただし、入力データまたは生成結果に関する第三者の権利（著作権、商標権、肖像権等）について、運営者は保証しません。必要な権利確認や各プラットフォームのルール確認はユーザーの責任で行ってください。</li>
                    <li>生成結果の利用によりユーザーまたは第三者に損害が生じた場合でも、運営者は運営者の故意または重大な過失がない限り責任を負いません。</li>
                  </ol>
                </section>

                <section>
                  <h3 className="text-lg font-bold mb-3" style={{ color: '#666' }}>
                    第9条（入力データの取り扱い）
                  </h3>
                  <ol className="list-decimal list-inside text-gray-700 space-y-2 ml-4">
                    <li>運営者は、入力データおよび生成結果（画像）を本サービス上に保存しません。</li>
                    <li>ただし、外部サービスへ処理のために送信される場合があり、外部サービス側での取り扱いは当該サービスの規約・ポリシーに従います。</li>
                  </ol>
                </section>

                <section>
                  <h3 className="text-lg font-bold mb-3" style={{ color: '#666' }}>
                    第10条（料金・決済）
                  </h3>
                  <ol className="list-decimal list-inside text-gray-700 space-y-2 ml-4">
                    <li>有料クレジットの決済は、決済事業者（Stripe）等の外部サービスを利用します。</li>
                    <li>決済に関する条件、手続、取り扱いは、当該外部サービスの規約等に従います。</li>
                  </ol>
                </section>

                <section>
                  <h3 className="text-lg font-bold mb-3" style={{ color: '#666' }}>
                    第11条（退会）
                  </h3>
                  <ol className="list-decimal list-inside text-gray-700 space-y-2 ml-4">
                    <li>ユーザーは、本サービスが定める方法により、いつでも退会（アカウント削除）できます。</li>
                    <li>退会すると本サービスの利用ができなくなります。</li>
                    <li>退会時点で未使用のクレジットがある場合、そのクレジットは失効します。</li>
                    <li>法令対応、会計・税務上の記録保持、不正利用防止等のため、運営者が一定期間保管する必要がある情報（購入履歴等）は、退会後も保持する場合があります。</li>
                  </ol>
                </section>

                <section>
                  <h3 className="text-lg font-bold mb-3" style={{ color: '#666' }}>
                    第12条（免責）
                  </h3>
                  <ol className="list-decimal list-inside text-gray-700 space-y-2 ml-4">
                    <li>運営者は、本サービスに事実上または法律上の瑕疵がないことを保証しません。</li>
                    <li>運営者は、本サービスに起因してユーザーに生じた損害について、運営者の故意または重大な過失がない限り責任を負いません。</li>
                    <li>運営者が責任を負う場合であっても、運営者の責任は、当該損害発生月にユーザーが運営者に支払った金額を上限とします（ただし、法令で別段の定めがある場合を除きます）。</li>
                  </ol>
                </section>

                <section>
                  <h3 className="text-lg font-bold mb-3" style={{ color: '#666' }}>
                    第13条（規約の変更）
                  </h3>
                  <p className="text-gray-700">
                    運営者は、必要に応じて本規約を変更できます。変更後の規約は、本サービス上での掲示その他運営者が適当と判断する方法で周知した時点から効力を生じます。
                  </p>
                </section>

                <section>
                  <h3 className="text-lg font-bold mb-3" style={{ color: '#666' }}>
                    第14条（準拠法・裁判管轄）
                  </h3>
                  <p className="text-gray-700">
                    本規約は日本法を準拠法とし、本サービスに関して紛争が生じた場合、福岡地方裁判所を専属的合意管轄とします。
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
