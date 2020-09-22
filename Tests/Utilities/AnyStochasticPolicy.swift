import OpenSpiel


/// Type-erased stochastic policy for a particular `Game`.
public struct AnyStochasticPolicy<Game: GameProtocol>: StochasticPolicy {
    public typealias Game = Game

    private let actionProbabilitiesClosure: (Game.State) -> [Game.Action : Double]

    public init<Policy>(_ policy: Policy) where Policy.Game == Game, Policy: StochasticPolicy {
        actionProbabilitiesClosure = policy.actionProbabilities
    }

    public func actionProbabilities(forState state: Game.State) -> [Game.Action : Double] {
        return actionProbabilitiesClosure(state)
    }
}
