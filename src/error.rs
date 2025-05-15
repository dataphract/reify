#[derive(Debug)]
pub enum Error {
    /// Reify made a mistake.
    Bug,
    /// The container's capacity is exhausted.
    Capacity,
    /// The operation timed out.
    TimedOut,
}
