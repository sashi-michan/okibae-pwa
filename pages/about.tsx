export default function About() {
  return (
    <div className="main-container">
      <div className="max-w-4xl mx-auto px-8 py-8">
        <div className="mb-6 text-center">
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
                minneやオンラインショップ、SNS投稿などにご自由にご活用いただけます🫶
              </p>
              <div className="bg-orange-50 border-l-4 border-orange-200 p-4 rounded-r-lg" style={{backgroundColor: '#EDBC9D20'}}>
                <p className="text-orange-600 text-sm" style={{color: '#B8899A'}}>
                  ※このアプリは現在 <strong>α版</strong>（2025年9月11日現在）です。<br/>
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
                <p className="typography-body">天気（晴れ・くもり・雨）を選ぶ</p>
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
            <div className="space-y-3 typography-body">
              <div className="flex items-start gap-2">
                <span className="text-brand-500">•</span>
                <p>1日に生成できる画像は現在「5枚まで」です</p>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-brand-500">•</span>
                <p>著作権のある画像は使用できません。ご自身で撮影した写真をお使いください</p>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-brand-500">•</span>
                <p>アップロードされた画像は、AIで処理するために外部の画像生成サービスに一時的に送信されます</p>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-brand-500">•</span>
                <p>生成画像は実際の色味・質感と異なる場合があります</p>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-brand-500">•</span>
                <p>不適切な内容、公序良俗に反する利用は禁止です</p>
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
                このアプリでは個人情報の収集は行っていません。<br/>
                アップロード画像は生成処理のために一時的に扱いますが、サーバー上に保存はされません。
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
