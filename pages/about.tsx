export default function About() {
  return (
    <div className="card">
      <h1 className="text-2xl font-bold">About</h1>
      <p className="mt-2 text-gray-700">
        これは OKIBAE（PWA版）のスターターです。<br/>
        画像を1枚選んで「背景ボタン」を押すと、まず仮合成→その後に背景消し＆色なじみ＆自然な影を自動で適用します。
      </p>
      <ul className="mt-4 list-disc pl-5 text-gray-700 space-y-1">
        <li>ホーム画面に追加できるPWA対応</li>
        <li>Tailwindで優しい見た目</li>
        <li>将来：テンプレやスライダー追加などを拡張</li>
      </ul>
    </div>
  )
}
