import { useRouter } from 'next/router'

export type ErrorType =
  | 'AUTH_REQUIRED'      // 401
  | 'INSUFFICIENT_CREDIT' // 403
  | 'INVALID_INPUT'      // 400
  | 'TIMEOUT'            // 504 or client timeout
  | 'SERVER_ERROR'       // 500
  | 'GENERATION_FAILED'  // AI generation failed (no image in response)
  | 'AUTH_RESTORE_FAILED' // Account restoration failed
  | 'AUTH_FETCH_FAILED'   // User data fetch failed
  | 'AUTH_SIGNOUT_FAILED' // Sign out failed
  | 'AUTH_OAUTH_FAILED'   // OAuth callback failed
  | 'AUTH_NO_CODE'        // OAuth code not found

export interface ErrorModalProps {
  isOpen: boolean
  onClose: () => void
  errorType: ErrorType
  requestId?: string
  creditConsumed?: boolean // true: 消費済み, false: 未消費, undefined: 不明
  customMessage?: string
}

const SUPPORT_EMAIL = 'okibae.help@gmail.com'

export function ErrorModal({
  isOpen,
  onClose,
  errorType,
  requestId,
  creditConsumed,
  customMessage
}: ErrorModalProps) {
  const router = useRouter()

  if (!isOpen) return null

  const getErrorContent = () => {
    switch (errorType) {
      case 'AUTH_REQUIRED':
        return {
          title: 'ログアウトしています',
          message: 'ログアウトしています。もう一度ログインしてください。',
          showRequestId: false,
          showSupport: false,
          primaryButton: {
            label: 'ログイン画面へ',
            action: () => router.replace('/login')
          }
        }

      case 'INSUFFICIENT_CREDIT':
        return {
          title: 'クレジットが不足しています',
          message: 'クレジットが不足しています。追加で購入後、生成いただけます。',
          showRequestId: false,
          showSupport: false,
          primaryButton: {
            label: '購入へ',
            action: () => router.replace('/purchase')
          }
        }

      case 'INVALID_INPUT':
        return {
          title: '画像を読み込めませんでした',
          message: customMessage || '画像を読み込めませんでした。別の画像でもう一度試してみてください。\n\n対応形式：JPG / PNG',
          showRequestId: true,
          showSupport: true,
          primaryButton: null
        }

      case 'TIMEOUT':
        return {
          title: '生成時間がタイムアウトしました',
          message: getCreditConsumedMessage(
            '生成時間がタイムアウトしました。もう一度生成ボタンを押してみてください',
            creditConsumed
          ),
          showRequestId: true,
          showSupport: true,
          primaryButton: null
        }

      case 'GENERATION_FAILED':
        return {
          title: '生成に失敗しました',
          message: getCreditConsumedMessage(
            '生成に失敗しました。もう一度生成ボタンを押してみてください',
            creditConsumed
          ),
          showRequestId: true,
          showSupport: true,
          primaryButton: null
        }

      case 'AUTH_RESTORE_FAILED':
        return {
          title: 'アカウント復活に失敗しました',
          message: 'アカウントの復活処理に失敗しました。時間を置いて再度ログインしてみてください。',
          showRequestId: false,
          showSupport: true,
          primaryButton: null
        }

      case 'AUTH_FETCH_FAILED':
        return {
          title: 'ユーザー情報の取得に失敗しました',
          message: 'ユーザー情報の取得に失敗しました。時間を置いて再度ログインしてみてください。',
          showRequestId: false,
          showSupport: true,
          primaryButton: null
        }

      case 'AUTH_SIGNOUT_FAILED':
        return {
          title: 'ログアウトに失敗しました',
          message: 'ログアウトに失敗しました。時間を置いて再度ログインしてみてください。',
          showRequestId: false,
          showSupport: true,
          primaryButton: null
        }

      case 'AUTH_OAUTH_FAILED':
        return {
          title: 'ログイン処理に失敗しました',
          message: 'ログイン処理に失敗しました。時間を置いて再度お試しください。',
          showRequestId: false,
          showSupport: true,
          primaryButton: null
        }

      case 'AUTH_NO_CODE':
        return {
          title: '認証コードが見つかりませんでした',
          message: '認証コードが見つかりませんでした。もう一度お試しください。',
          showRequestId: false,
          showSupport: true,
          primaryButton: null
        }

      case 'SERVER_ERROR':
      default:
        return {
          title: 'サーバー側でエラーが発生しました',
          message: getCreditConsumedMessage(
            customMessage || 'サーバー側でエラーが発生しました。もう一度生成ボタンを押してみてください',
            creditConsumed
          ),
          showRequestId: true,
          showSupport: true,
          primaryButton: null
        }
    }
  }

  const getCreditConsumedMessage = (baseMessage: string, consumed?: boolean) => {
    if (consumed === false) {
      return `${baseMessage}（クレジットは消費されていません）。`
    }
    if (consumed === true) {
      return `${baseMessage}（クレジットが消費されています）。`
    }
    // undefined: 不明
    return `${baseMessage}（通常はクレジットは消費されません。もし消費されていた場合は、リクエストIDを添えてご連絡ください）。`
  }

  const getMailtoLink = () => {
    const subject = encodeURIComponent('OKIBAE エラー報告')
    const body = requestId ? encodeURIComponent(`リクエストID: ${requestId}\n\nエラー内容:\n`) : ''
    return `mailto:${SUPPORT_EMAIL}?subject=${subject}${body ? `&body=${body}` : ''}`
  }

  const content = getErrorContent()

  return (
    <div
      className="fixed inset-0 bg-black/40 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-3xl shadow-2xl max-w-md w-full p-8 relative"
        onClick={(e) => e.stopPropagation()}
      >
        {/* 右上の×ボタン */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-gray-600 transition-colors"
          aria-label="閉じる"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>

        {/* タイトル */}
        <h2 className="text-xl font-medium text-gray-800 mb-4 text-center">
          {content.title}
        </h2>

        {/* 本文 */}
        <p className="text-sm text-gray-600 leading-relaxed whitespace-pre-line mb-6">
          {content.message}
        </p>

        {/* リクエストID */}
        {content.showRequestId && requestId && (
          <div className="mb-4 p-3 bg-gray-50 rounded-xl border border-gray-200">
            <p className="text-xs text-gray-500 mb-1">リクエストID</p>
            <p className="text-xs text-gray-700 font-mono break-all">{requestId}</p>
          </div>
        )}

        {/* サポートメール */}
        {content.showSupport && (
          <p className="text-xs text-gray-500 mb-6 text-center">
            何度か試しても失敗する場合、この画面のスクリーンショットを運営に送ってください。
            <br />
            <a
              href={getMailtoLink()}
              className="text-[#C2A2A8] underline hover:text-[#C792A3] transition-colors"
            >
              {SUPPORT_EMAIL}
            </a>
          </p>
        )}

        {/* ボタン（401/403のみ表示） */}
        {content.primaryButton && (
          <button
            onClick={content.primaryButton.action}
            className="w-full bg-[#C792A3] hover:bg-[#C2A2A8] text-white font-medium py-3 px-6 rounded-full transition-all duration-300 text-sm"
          >
            {content.primaryButton.label}
          </button>
        )}
      </div>
    </div>
  )
}
