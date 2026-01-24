import { useRouter } from 'next/router'
import Link from 'next/link'

export default function About() {
  const router = useRouter()

  return (
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
          {/* このアプリについて */}
          <div className="card animate-slide-up">
            <h2 className="typography-step-title mb-4">
              このアプリについて
            </h2>
            <div className="space-y-4 typography-body text-base leading-relaxed">
              <p>
                このアプリは、ハンドメイド作品やお気に入りの小物を、簡単に「置き画」風の写真に仕上げられるツールです。<br/>
                背景や光の雰囲気を選ぶだけで、AIが自動でおしゃれな画像を生成します。
              </p>
              <p>
                minneなどのオンラインショップ、SNS投稿などにご自由にご活用いただけます。<br />
                商用利用の際は、下の注意事項をよくご確認ください。
              </p>
              <div className="bg-orange-50 border-l-4 border-orange-200 p-4 rounded-r-lg" style={{backgroundColor: '#EDBC9D20'}}>
                <p className="text-orange-600 text-sm" style={{color: '#B8899A'}}>
                  ※このアプリは現在 <strong>β版</strong>（2026年1月7日現在）です。<br/>
                  機能や仕様は今後変更されることがあります。
                </p>
              </div>
            </div>
          </div>

          {/* 使い方 */}
          <div className="card animate-slide-up">
            <h2 className="typography-step-title mb-4">
              使い方
            </h2>
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <span className="flex-shrink-0 w-6 h-6 bg-brand-500 text-white rounded-full text-xs font-bold flex items-center justify-center">1</span>
                <p className="typography-body">手元の写真をアップロード</p>
              </div>
              <div className="flex items-start gap-3">
                <span className="flex-shrink-0 w-6 h-6 bg-brand-500 text-white rounded-full text-xs font-bold flex items-center justify-center">2</span>
                <p className="typography-body">背景スタイルを選ぶ</p>
              </div>
              <div className="flex items-start gap-3">
                <span className="flex-shrink-0 w-6 h-6 bg-brand-500 text-white rounded-full text-xs font-bold flex items-center justify-center">3</span>
                <p className="typography-body">天気（晴れ・くもり・雨）を選ぶ<br />光の当たり方や空気感が変わります</p>
              </div>
              <div className="flex items-start gap-3">
                <span className="flex-shrink-0 w-6 h-6 bg-brand-500 text-white rounded-full text-xs font-bold flex items-center justify-center">4</span>
                <p className="typography-body">「生成する」ボタンを押す</p>
              </div>
              <div className="flex items-start gap-3">
                <span className="flex-shrink-0 w-6 h-6 bg-brand-500 text-white rounded-full text-xs font-bold flex items-center justify-center">5</span>
                <p className="typography-body">「保存」ボタンでスマホやPCに保存！</p>
              </div>
            </div>
            <div className="mt-6 bg-orange-50 border-l-4 border-orange-200 p-4 rounded-r-lg" style={{backgroundColor: '#EDBC9D20'}}>
              <p className="text-sm" style={{color: '#B8899A'}}>
                💡 同じ設定でも、何度か試してみると結果が変わることもあります。<br/>
                お気に入りの結果が出るまで試してみると良いかもしれません！
              </p>
            </div>
          </div>

          {/* 利用上の注意 */}
          <div className="card animate-slide-up">
            <h2 className="typography-step-title mb-4">
              利用上の注意
            </h2>
            <div className="space-y-6 typography-body">
              {/* 1. 著作権と「使ってよい写真」について */}
              <div>
                <h3 className="font-bold mb-2" style={{ color: '#666' }}>1. 著作権と「使ってよい写真」について</h3>
                <p className="mb-2">
                  アップロードする写真は、ご自身で撮影したものや、権利者から許可を得ているものを使ってください。<br />
                  以下のような画像の使用はお控えください（法律で禁止されている場合があります）。
                </p>
                <ul className="ml-4 space-y-1">
                  <li>・❌ ネットで拾った画像（他人が撮った写真）</li>
                  <li>・❌ 漫画・アニメのキャラクターや、有名人が写っている画像※</li>
                  <li>・❌ 他の作家さんの作品画像（許可なく加工すること）</li>
                </ul>
                <p className="text-sm mt-2" style={{ color: '#666' }}>※キャラクターグッズなどをご自身で撮影されたものであればご利用いただけます</p>
              </div>

              {/* 2. 「AI生成画像であること」の記載を推奨しています */}
              <div>
                <h3 className="font-bold mb-2" style={{ color: '#666' }}>2. 「AI生成画像であること」の記載を推奨しています</h3>
                <p className="mb-2">
                  本アプリはAI（人工知能）を使って背景を描き出しています。<br />
                  フリマアプリやSNSに投稿する際は、見た人が「実物もこの背景で撮ったのかな？」と誤解しないよう、<br />
                  キャプションなどに一言添えていただくことをおすすめします。
                </p>
                <div className="bg-orange-50 border-l-4 border-orange-200 p-3 rounded-r-lg mt-2" style={{backgroundColor: '#EDBC9D20'}}>
                  <p className="text-sm" style={{ color: '#B8899A' }}>
                    <strong>おすすめの書き方例：</strong><br />
                    ・「※背景はAIによって作成したイメージです」<br />
                    ・「※背景は演出として合成しています」
                  </p>
                </div>
              </div>

              {/* 3. AIの「うっかり」にご注意ください */}
              <div>
                <h3 className="font-bold mb-2" style={{ color: '#666' }}>3. AIの「うっかり」にご注意ください</h3>
                <p className="mb-2">
                  商品自体が変形・変更などされないように調整していますが、まれに不思議な画像が作られてしまうことがあります。<br />
                  念のため、保存する前に必ず<strong>「商品の見た目や魅力が正しく伝わっているか」</strong>をご自身の目でチェックしてください。
                </p>
                <p className="text-sm" style={{ color: '#666' }}>
                  フリマアプリで利用する場合、商品の傷や汚れが消えてしまうとトラブルの元となる恐れがありますのでご注意ください。
                </p>
              </div>

              {/* 4. 商用利用について */}
              <div>
                <h3 className="font-bold mb-2" style={{ color: '#666' }}>4. 商用利用について</h3>
                <p className="mb-2">
                  作成した画像は、ECサイト、SNS、チラシなどで商用利用（お仕事での利用）OKです！<br />
                  ただし、その画像を使って起きたトラブル（例：「実物と違う！」というクレームなど）については、<br />
                  ユーザー様ご自身の責任で対応をお願いいたします。
                </p>
              </div>

              {/* 利用規約リンク */}
              <div className="mt-4 pt-4 border-t border-pink-100">
                <p className="text-sm" style={{ color: '#666' }}>
                  詳しくは<Link href="/terms" className="text-brand-600 hover:text-brand-700 underline">利用規約</Link>をご確認ください。
                </p>
              </div>
            </div>
          </div>

          {/* データの取扱いについて */}
          <div className="card animate-slide-up">
            <h2 className="typography-step-title mb-4">
              データの取扱いについて
            </h2>
            <div className="space-y-4 typography-body">
              <p>
                本アプリでは、皆様の大切な作品画像を守るため、以下の仕組みで運用しています。
              </p>

              <div>
                <h3 className="font-bold mb-2" style={{ color: '#666' }}>・画像は保存されません</h3>
                <p>
                  アップロードされた画像は、AIによる背景生成処理を行うためだけに使用し、アプリ（サーバー）内に保存・蓄積することはありません。<br />
                  処理が終わり次第、データはメモリから消去されます。
                </p>
              </div>

              <div>
                <h3 className="font-bold mb-2" style={{ color: '#666' }}>・AIの学習には使われません</h3>
                <p>
                  画像生成エンジンには、セキュリティ強度の高い「Google Vertex AI」を採用しています。ここで処理される画像データが、AIの学習（トレーニング）に勝手に利用されることはありません。
                </p>
              </div>

              <div className="mt-4 pt-4 border-t border-pink-100">
                <p className="text-sm" style={{ color: '#666' }}>
                  詳しくは<Link href="/privacy" className="text-brand-600 hover:text-brand-700 underline">プライバシーポリシー</Link>をご確認ください。
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
