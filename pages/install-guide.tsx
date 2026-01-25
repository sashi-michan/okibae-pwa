import { useRouter } from 'next/router'
import { useState, useEffect } from 'react'

export default function InstallGuide() {
  const router = useRouter()
  const [iosVersion, setIosVersion] = useState<number | null>(null)

  useEffect(() => {
    // iOSバージョン検出
    const ua = navigator.userAgent
    const match = ua.match(/OS (\d+)_/)
    if (match) {
      setIosVersion(parseInt(match[1], 10))
    }
  }, [])

  // iOS 18以降かどうか
  const isModernIOS = iosVersion === null || iosVersion >= 18

  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-50 via-cream-50 to-orange-50">
      <div className="max-w-2xl mx-auto px-8 py-8">
        {/* ヘッダー */}
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

          <button
            onClick={() => router.push('/')}
            className="flex items-center justify-center gap-3 w-full hover:opacity-80 transition-opacity"
          >
            <img
              src="/okibae-icon.svg"
              alt="OKIBAE"
              className="h-10 w-10"
            />
            <h1 className="typography-main-title">OKIBAE</h1>
          </button>
        </div>

        {/* iOS用手順 */}
        <div className="card animate-slide-up">
          <h2 className="typography-step-title mb-6">ホーム画面に追加する（iPhone・iPad）</h2>

          <div className="space-y-6">
            {/* Step 1 - iOS バージョンで分岐 */}
            {isModernIOS ? (
              // iOS 18以降: 右下の … メニュー
              <div className="flex gap-4">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-brand-500 text-white flex items-center justify-center font-semibold">
                  1
                </div>
                <div className="flex-1">
                  <h3 className="font-medium text-gray-800 mb-2">右下の<span className="inline-block px-1.5 py-0.5 bg-gray-100 rounded text-xs">…</span>をタップ</h3>
                  <p className="text-sm text-gray-600 mb-3">
                    右下にメニュー <span className="inline-block px-1.5 py-0.5 bg-gray-100 rounded text-xs">…</span> が表示されます。これをタップしてください
                  </p>
                  <div className="bg-orange-50 border-l-4 border-orange-200 p-3 rounded-r-lg" style={{ backgroundColor: 'rgba(237, 188, 157, 0.125)' }}>
                    <p className="text-orange-600 text-sm" style={{ color: 'rgb(184, 137, 154)' }}>
                      💡 <strong>ポイント：</strong>画面を少し下にスクロールすると、画面下部のメニューが表示されます
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              // iOS 17以前: 画面下部の共有ボタン
              <div className="flex gap-4">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-brand-500 text-white flex items-center justify-center font-semibold">
                  1
                </div>
                <div className="flex-1">
                  <h3 className="font-medium text-gray-800 mb-2">
                    画面下部の共有ボタン
                    <span className="inline-flex items-center justify-center px-1.5 py-0.5 bg-gray-100 rounded text-xs ml-1">
                      <svg width="12" height="14" viewBox="0 0 12 14" fill="none" xmlns="http://www.w3.org/2000/svg" className="inline-block">
                        <path d="M6 0.5L6 9.5M6 0.5L3 3.5M6 0.5L9 3.5M1 8.5L1 12.5C1 12.7761 1.22386 13 1.5 13L10.5 13C10.7761 13 11 12.7761 11 12.5L11 8.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    </span>
                    をタップ
                  </h3>
                  <p className="text-sm text-gray-600 mb-3">
                    画面下部の中央に共有ボタン（四角に上向き矢印のアイコン）が表示されます。これをタップしてください
                  </p>
                  <div className="bg-orange-50 border-l-4 border-orange-200 p-3 rounded-r-lg" style={{ backgroundColor: 'rgba(237, 188, 157, 0.125)' }}>
                    <p className="text-orange-600 text-sm" style={{ color: 'rgb(184, 137, 154)' }}>
                      💡 <strong>ポイント：</strong>画面を少し下にスクロールすると、画面下部のメニューが表示されます
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Step 2-3: iOS 18以降のみ表示 */}
            {isModernIOS && (
              <>
                {/* Step 2 */}
                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-brand-500 text-white flex items-center justify-center font-semibold">
                    2
                  </div>
                  <div className="flex-1">
                    <h3 className="font-medium text-gray-800 mb-2">「共有」をタップ</h3>
                    <p className="text-sm text-gray-600 mb-3">
                      メニューの中から「共有」アイコンをタップしてください
                    </p>
                    <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                      <img
                        src="/install-guide/ios-step1.png"
                        alt="共有をタップ"
                        className="w-full rounded"
                        onError={(e) => {
                          e.currentTarget.style.display = 'none'
                        }}
                      />
                    </div>
                  </div>
                </div>

                {/* Step 3 */}
                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-brand-500 text-white flex items-center justify-center font-semibold">
                    3
                  </div>
                  <div className="flex-1">
                    <h3 className="font-medium text-gray-800 mb-2">右下の「その他」をタップ</h3>
                    <p className="text-sm text-gray-600 mb-3">
                      共有メニューの右下にある「その他」をタップしてください
                    </p>
                    <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                      <img
                        src="/install-guide/ios-step2.png"
                        alt="その他をタップ"
                        className="w-full rounded"
                        onError={(e) => {
                          e.currentTarget.style.display = 'none'
                        }}
                      />
                    </div>
                  </div>
                </div>
              </>
            )}

            {/* Step 4 (iOS 18: Step 4, iOS 17以前: Step 2) */}
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-brand-500 text-white flex items-center justify-center font-semibold">
                {isModernIOS ? '4' : '2'}
              </div>
              <div className="flex-1">
                <h3 className="font-medium text-gray-800 mb-2">「ホーム画面に追加」をタップ</h3>
                <p className="text-sm text-gray-600 mb-3">
                  メニューの中から「ホーム画面に追加」を選択してください
                </p>
                <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                  <img
                    src="/install-guide/ios-step3.png"
                    alt="ホーム画面に追加をタップ"
                    className="w-full rounded"
                    onError={(e) => {
                      e.currentTarget.style.display = 'none'
                    }}
                  />
                </div>
              </div>
            </div>

            {/* Step 5 (iOS 18: Step 5, iOS 17以前: Step 3) */}
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-brand-500 text-white flex items-center justify-center font-semibold">
                {isModernIOS ? '5' : '3'}
              </div>
              <div className="flex-1">
                <h3 className="font-medium text-gray-800 mb-2">「追加」をタップ</h3>
                <p className="text-sm text-gray-600 mb-3">
                  右上の「追加」ボタンをタップすると、ホーム画面にOKIBAEのアイコンが追加されます
                </p>
                <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                  <img
                    src="/install-guide/ios-step4.png"
                    alt="追加をタップ"
                    className="w-full rounded"
                    onError={(e) => {
                      e.currentTarget.style.display = 'none'
                    }}
                  />
                </div>
              </div>
            </div>

            {/* 完了メッセージ */}
            <div className="bg-orange-50 border-l-4 border-orange-200 p-4 rounded-r-lg mt-6" style={{ backgroundColor: 'rgba(237, 188, 157, 0.125)' }}>
              <p className="text-orange-600 text-sm" style={{ color: 'rgb(184, 137, 154)' }}>
                ✨ 次回からはホーム画面のアイコンをタップするだけで、すぐにOKIBAEを使えます！
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
