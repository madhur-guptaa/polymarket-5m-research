/** Panel — the card primitive. Flex column so a body child can fill height. */
export default function Panel({ title, right, className = "", children }) {
  return (
    <section className={`flex flex-col rounded-lg border border-border bg-panel ${className}`}>
      {title && (
        <header className="flex shrink-0 items-center justify-between border-b border-border px-3 py-2">
          <span className="text-[11px] uppercase tracking-[0.15em] text-muted">{title}</span>
          {right}
        </header>
      )}
      <div className="min-h-0 flex-1 p-3">{children}</div>
    </section>
  );
}
