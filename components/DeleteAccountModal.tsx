import { useState } from 'react'

interface DeleteAccountModalProps {
  isOpen: boolean
  onClose: () => void
  onConfirm: () => Promise<void>
  balance: number
}

export default function DeleteAccountModal({
  isOpen,
  onClose,
  onConfirm,
  balance
}: DeleteAccountModalProps) {
  const [isDeleting, setIsDeleting] = useState(false)

  if (!isOpen) return null

  const handleConfirm = async () => {
    setIsDeleting(true)
    try {
      await onConfirm()
    } catch (error) {
      console.error('退会処理エラー:', error)
      alert('退会処理に失敗しました。時間をおいて再度お試しください。')
    } finally {
      setIsDeleting(false)
    }
  }

  return (
    <div
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-[9999] p-4 overflow-y-auto"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-lg shadow-xl max-w-md w-full p-6 my-8"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="text-xl font-bold mb-4" style={{ color: '#666' }}>
          退会の確認
        </h2>

        <div className="space-y-3 text-sm text-gray-700 mb-6">
          <p className="font-medium">この操作は取り消せません。本当に退会しますか？</p>

          <div className="bg-red-50 border border-red-200 rounded-lg p-3 space-y-2">
            <p className="font-medium text-red-800">退会すると以下の情報が削除されます：</p>
            <ul className="list-disc list-inside space-y-1 text-red-700">
              <li>アカウント情報</li>
              <li>クレジット残高（現在: {balance}回）</li>
              <li>すべての利用履歴</li>
            </ul>
          </div>

          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 space-y-2">
            <p className="font-medium text-yellow-800">注意事項：</p>
            <ul className="list-disc list-inside space-y-1 text-yellow-700">
              <li>退会後の情報復元はできません</li>
              <li>再登録時、初回特典（5回分無料）は付与されません</li>
            </ul>
          </div>
        </div>

        <div className="flex gap-3">
          <button
            onClick={onClose}
            disabled={isDeleting}
            className="flex-1 px-4 py-3 bg-gray-100 text-gray-700 rounded-lg font-medium hover:bg-gray-200 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            キャンセル
          </button>
          <button
            onClick={handleConfirm}
            disabled={isDeleting}
            className="flex-1 px-4 py-3 text-white rounded-lg font-medium transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            style={{
              backgroundColor: '#C792A3',
            }}
            onMouseEnter={(e) => {
              if (!isDeleting) {
                e.currentTarget.style.backgroundColor = '#B07D91'
              }
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = '#C792A3'
            }}
          >
            {isDeleting ? '処理中...' : '退会する'}
          </button>
        </div>
      </div>
    </div>
  )
}
