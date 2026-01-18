/**
 * ロガーユーティリティ
 * 開発環境と本番環境でログ出力を制御
 */

const isDev = process.env.NODE_ENV === 'development'

export const logger = {
  /**
   * 開発環境専用ログ
   * 本番環境では出力されない
   */
  dev: (...args: any[]) => {
    if (isDev) console.log(...args)
  },

  /**
   * 情報ログ（本番環境でも出力）
   * 処理開始・完了などの正常系イベント
   */
  info: (...args: any[]) => {
    console.log(...args)
  },

  /**
   * 警告ログ（本番環境でも出力）
   * クレジット不足など、エラーではないが注意が必要な状況
   */
  warn: (...args: any[]) => {
    console.warn(...args)
  },

  /**
   * エラーログ（本番環境でも出力）
   * 認証失敗、API呼び出し失敗などの異常系
   */
  error: (...args: any[]) => {
    console.error(...args)
  },
}

/**
 * リクエストIDを生成
 * 使用例: const requestId = `ai-shadows:${generateRequestId()}`
 */
export function generateRequestId(): string {
  return `${Date.now()}_${Math.random().toString(36).slice(2, 11)}`
}
