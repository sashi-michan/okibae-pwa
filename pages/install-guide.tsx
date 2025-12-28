import { useRouter } from 'next/router'

export default function InstallGuide() {
  const router = useRouter()

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
            {/* Step 1 */}
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-brand-500 text-white flex items-center justify-center font-semibold">
                1
              </div>
              <div className="flex-1">
                <h3 className="font-medium text-gray-800 mb-2">画面下の共有ボタンをタップ</h3>
                <p className="text-sm text-gray-600 mb-3">
                  Safari下部の<span className="inline-flex items-center mx-1">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M18 16.08c-.76 0-1.44.3-1.96.77L8.91 12.7c.05-.23.09-.46.09-.7s-.04-.47-.09-.7l7.05-4.11c.54.5 1.25.81 2.04.81 1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3c0 .24.04.47.09.7L8.04 9.81C7.5 9.31 6.79 9 6 9c-1.66 0-3 1.34-3 3s1.34 3 3 3c.79 0 1.5-.31 2.04-.81l7.12 4.16c-.05.21-.08.43-.08.65 0 1.61 1.31 2.92 2.92 2.92 1.61 0 2.92-1.31 2.92-2.92s-1.31-2.92-2.92-2.92z"/>
                    </svg>
                  </span>ボタンをタップしてください
                </p>
                <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                  <img
                    src="/install-guide/ios-step1.png"
                    alt="共有ボタンをタップ"
                    className="w-full rounded"
                    onError={(e) => {
                      e.currentTarget.style.display = 'none'
                    }}
                  />
                </div>
              </div>
            </div>

            {/* Step 2 */}
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-brand-500 text-white flex items-center justify-center font-semibold">
                2
              </div>
              <div className="flex-1">
                <h3 className="font-medium text-gray-800 mb-2">「ホーム画面に追加」を選択</h3>
                <p className="text-sm text-gray-600 mb-3">
                  メニューの中から「ホーム画面に追加」をタップしてください
                </p>
                <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                  <img
                    src="/install-guide/ios-step2.png"
                    alt="ホーム画面に追加を選択"
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
                <h3 className="font-medium text-gray-800 mb-2">「追加」をタップ</h3>
                <p className="text-sm text-gray-600 mb-3">
                  右上の「追加」ボタンをタップすると、ホーム画面にOKIBAEのアイコンが追加されます
                </p>
                <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                  <img
                    src="/install-guide/ios-step3.png"
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
