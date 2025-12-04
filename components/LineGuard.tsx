import { useEffect, useState } from 'react';

export const LineGuard = () => {
  const [isTrapped, setIsTrapped] = useState(false);
  const [osType, setOsType] = useState<'ios' | 'android' | 'other'>('other');

  useEffect(() => {
    // 1. ユーザーエージェント（ブラウザの正体）を取得
    const ua = navigator.userAgent.toLowerCase();
    
    // 2. LINEまたはInstagramなどのアプリ内ブラウザか判定
    const isInAppBrowser = ua.includes('line') || ua.includes('instagram') || ua.includes('facebook');
    
    // 3. OSを判定（iPhoneかAndroidかでお母さんへの案内を変えるため）
    if (ua.includes('iphone') || ua.includes('ipad')) {
      setOsType('ios');
    } else if (ua.includes('android')) {
      setOsType('android');
    }

    // 4. アプリ内ブラウザなら「閉じ込めモード」オン！
    if (isInAppBrowser) {
      setIsTrapped(true);
      // スクロールできないようにbodyを固定
      document.body.style.overflow = 'hidden';
    }
  }, []);

  // 閉じ込められてなければ何も表示しない
  if (!isTrapped) return null;

  return (
    // 画面全体を覆うオーバーレイ（z-50で最前面に！）
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-gradient-to-br from-pink-50 via-orange-50 to-orange-100 p-6 text-center">

      {/* 注意喚起タイトル */}
      <h2 className="text-xl font-bold mb-6" style={{ color: '#C2A2A8' }}>
        ブラウザを変更してください
      </h2>

      {/* ここにGIFを入れるエリア */}
      <div className="w-64 h-64 bg-white/80 backdrop-blur-sm rounded-2xl flex items-center justify-center mb-6 border-2 border-dashed shadow-sm" style={{ borderColor: '#C2A2A8' }}>
        <p className="text-sm" style={{ color: '#C2A2A8' }}>
          ここに<br/>
          {osType === 'ios' ? 'iPhone用' : 'Android用'}の<br/>
          GIFアニメを入れる
        </p>
      </div>

      {/* 操作説明テキスト */}
      <div className="space-y-4 max-w-sm">
        <p className="text-gray-700">
          LINEなどのアプリ内ブラウザでは<br/>
          正常に動作しない場合があります。
        </p>

        <div className="bg-white/90 backdrop-blur-sm p-5 rounded-2xl shadow-md border" style={{ borderColor: '#C2A2A8' }}>
          <p className="font-bold mb-3" style={{ color: '#C792A3' }}>
            解決方法
          </p>
          <p className="text-sm text-gray-700 leading-relaxed">
            {osType === 'ios' ? (
              // iPhone向けの説明
              <>
                1. 共有ボタン <span className="inline-block px-1.5 py-0.5 bg-gray-100 rounded text-xs">↑</span> をタップ<br/>
                2. <b>「Safariで開く」</b>を選択
              </>
            ) : (
              // Android向けの説明
              <>
                1. 右上のメニュー <span className="inline-block px-1.5 py-0.5 bg-gray-100 rounded text-xs">︙</span> をタップ<br/>
                2. <b>「ブラウザで開く」</b>を選択
              </>
            )}
          </p>
        </div>
      </div>
    </div>
  );
};