import Link from 'next/link';
import { useEffect, useState } from 'react';

export default function NavBar() {
  const [dailyUsage, setDailyUsage] = useState({ count: 0, date: '' })

  // 日次使用制限の管理
  useEffect(() => {
    const initDailyUsage = () => {
      const today = new Date().toISOString().split('T')[0] // YYYY-MM-DD
      const savedDate = localStorage.getItem('okibae-date')
      const savedCount = localStorage.getItem('okibae-count')
      
      console.log('NavBar init:', { today, savedDate, savedCount })
      
      if (savedDate === today && savedCount) {
        // 今日のデータがある場合
        const count = parseInt(savedCount, 10)
        console.log('NavBar loading existing count:', count)
        setDailyUsage({ count, date: today })
      } else {
        // 初回または日付が変わった場合はリセット
        console.log('NavBar resetting count to 0')
        localStorage.setItem('okibae-date', today)
        localStorage.setItem('okibae-count', '0')
        setDailyUsage({ count: 0, date: today })
      }
    }
    
    initDailyUsage()

    // localStorageの変更を監視
    const handleStorageChange = () => {
      console.log('Storage change detected, updating usage...')
      initDailyUsage()
    }

    // カスタムイベントを監視（同一ページ内での更新用）
    window.addEventListener('okibae-usage-update', handleStorageChange)

    return () => {
      window.removeEventListener('storage', handleStorageChange)
      window.removeEventListener('okibae-usage-update', handleStorageChange)
    }
  }, [])

  return (
    <header className="sticky top-0 z-10 bg-gradient-to-r from-pink-50/95 via-white/90 to-orange-50/95 backdrop-blur-md supports-[backdrop-filter]:bg-white/80 border-b border-pink-100/50 shadow-sm">
      <nav className="mx-auto max-w-3xl px-4 py-3 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2 font-bold transition-all duration-300 hover:scale-105">
          <img 
            src="/okibae-icon.svg" 
            alt="OKIBAE" 
            className="h-8 w-8 transition-transform duration-300 hover:scale-110" 
          />
        </Link>
        <div className="flex items-center gap-4 text-sm">
          <span className="px-2 py-1 rounded-full text-xs font-medium" style={{
            backgroundColor: dailyUsage.count >= 5 ? '#FEE2E2' : 
                           dailyUsage.count >= 3 ? '#FED7AA' : 
                           '#C792A3',
            color: dailyUsage.count >= 5 ? '#A0616A' : 
                   dailyUsage.count >= 3 ? '#C4894D' : 
                   'white'
          }}>
            のこり {Math.max(0, 5 - dailyUsage.count)}/5
          </span>
          <Link href="/about" className="hover:text-brand-600 transition-colors duration-200 relative group">
            使い方
            <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-brand-500 group-hover:w-full transition-all duration-300"></span>
          </Link>
        </div>
      </nav>
    </header>
  )
}
