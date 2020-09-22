import TensorFlow
import OpenSpiel

import Utilities


/// Random Bandit algorithm.
public typealias RandomGambler = UniformRandomPolicy<MultiArmedBandit>


/// Simple Bandit algorithm with true reward averaging per Chapter 2.4
public struct AveragingGambler: StochasticPolicy {
    public typealias Game = MultiArmedBandit
    
    /// Instance of the Multi-armed Bandit game being played
    private let game: Game

    /// Action exploration rate.
    private var ε: Double
    
    /// Number of times a state has been visited
    private var N: [Game.Action: Int]
    
    /// Estimated value of each action.
    private var Q: [Game.Action: Double]
    
    public init(_ game: Game, ε: Double = 0.01) {
        self.game = game
        self.ε = ε

        let initialN = Array(repeating: 0, count: game.allActions.count)
        N = Dictionary(uniqueKeysWithValues: zip(game.allActions, initialN))

        let initialQ = Tensor(randomUniform: TensorShape([game.allActions.count]),
                              lowerBound: Tensor(0.0),
                              upperBound: Tensor(1e-5))
        Q = Dictionary(uniqueKeysWithValues: zip(game.allActions, initialQ.scalars))
    }
    
    public func actionProbabilities(forState state: Game.State) -> [Game.Action: Double] {
        assert(!state.isTerminal)
        assert(!state.legalActions.isEmpty)
        
        let legalQValues = state.legalActions.map{ Q[$0]! }
        var legalActionProbabilities = Array(repeating: ε / Double(state.legalActions.count - 1),
                                             count: state.legalActions.count)
        legalActionProbabilities[legalQValues.argMax()!] = 1 - ε

        return Dictionary(uniqueKeysWithValues: zip(state.legalActions, legalActionProbabilities))
    }
    
    public mutating func update(with action: Game.Action, reward: Double) {
        N[action] = N[action]! + 1
        Q[action] = Q[action]! + (reward - Q[action]!) / Double(N[action]!)
    }
}


/// Simple Bandit algorithm with fixed learning step per Chapter 2.5
public struct EpsilonGreedyGambler: StochasticPolicy {
    public typealias Game = MultiArmedBandit
    
    /// Instance of the Multi-armed Bandit game being played
    private let game: Game

    /// Action exploration rate
    private var ε: Double
    
    /// Value estimation update rate
    private var α: Double
    
    /// Estimated value of each action.
    private var Q: [Game.Action: Double]
    
    public init(_ game: Game, ε: Double = 0.01, α: Double = 0.1) {
        self.game = game
        self.ε = ε
        self.α = α

        let initialQ = Tensor(randomUniform: TensorShape([game.allActions.count]),
                              lowerBound: Tensor(0.0),
                              upperBound: Tensor(1e-5))
        Q = Dictionary(uniqueKeysWithValues: zip(game.allActions, initialQ.scalars))
    }
    
    public func actionProbabilities(forState state: Game.State) -> [Game.Action : Double] {
        assert(!state.isTerminal)
        assert(!state.legalActions.isEmpty)
        
        let legalQValues = state.legalActions.map{ Q[$0]! }
        var legalActionProbabilities = Array(repeating: ε / Double(state.legalActions.count - 1),
                                             count: state.legalActions.count)
        legalActionProbabilities[legalQValues.argMax()!] = 1 - ε

        return Dictionary(uniqueKeysWithValues: zip(state.legalActions, legalActionProbabilities))
    }
    
    public mutating func update(with action: Game.Action, reward: Double) {
        Q[action] = Q[action]! + α * (reward - Q[action]!)
    }
}


/// Optimistic Initial Values algorithm per Chapter 2.6
public struct OptimisticGambler: StochasticPolicy {
    public typealias Game = MultiArmedBandit
    
    /// Instance of the Multi-armed Bandit game being played
    private let game: Game

    /// Action exploration rate
    private var ε: Double
    
    /// Value estimation update rate
    private var α: Double
    
    /// Estimated value of each action.
    private var Q: [Game.Action: Double]
    
    public init(_ game: Game, ε: Double = 0.01, α: Double = 0.1, Q0:Double = 10.0) {
        self.game = game
        self.ε = ε
        self.α = α

        let initialQ = Tensor(randomUniform: TensorShape([game.allActions.count]),
                              lowerBound: Tensor(0.0),
                              upperBound: Tensor(Q0)).scalars
        Q = Dictionary(uniqueKeysWithValues: zip(game.allActions, initialQ))
    }
    
    public func actionProbabilities(forState state: Game.State) -> [Game.Action : Double] {
        assert(!state.isTerminal)
        assert(!state.legalActions.isEmpty)
        
        let legalQValues = state.legalActions.map{ Q[$0]! }
        var legalActionProbabilities = Array(repeating: ε / Double(state.legalActions.count - 1),
                                             count: state.legalActions.count)
        legalActionProbabilities[legalQValues.argMax()!] = 1 - ε

        return Dictionary(uniqueKeysWithValues: zip(state.legalActions, legalActionProbabilities))
    }
    
    public mutating func update(with action: Game.Action, reward: Double) {
        Q[action] = Q[action]! + α * (reward - Q[action]!)
    }
}


/// Upper Confidence Bound Action Selection algorithm per Chapter 2.7
public struct UCBGambler: DeterministicPolicy {
    public typealias Game = MultiArmedBandit
    
    /// Instance of the Multi-armed Bandit game being played
    private let game: Game

    /// UCB exploration-exploitation tuning constant
    private var c: Double
    
    /// Number of times a state has been visited
    private var N: [Game.Action: Int]
    
    /// Estimated value of each action.
    private var Q: [Game.Action: Double]
    
    /// Upper confidence bound of each action.
    private var UCB: [Game.Action: Double] {
        let t = N.values.reduce(0, +) + 1

        var UCB = Q
        for (action, q) in Q {
            let n = N[action]!
            UCB[action] = q + c + sqrt(log(Double(t)) / (Double(n) + 1e-5))
        }
        
        return UCB
    }
    
    public init(_ game: Game, c: Double = 1.0) {
        self.game = game
        self.c = c

        let initialN = Array(repeating: 0, count: game.allActions.count)
        N = Dictionary(uniqueKeysWithValues: zip(game.allActions, initialN))

        let initialQ = Tensor(randomUniform: TensorShape([game.allActions.count]),
                              lowerBound: Tensor(0.0),
                              upperBound: Tensor(1e-5)).scalars
        Q = Dictionary(uniqueKeysWithValues: zip(game.allActions, initialQ))
    }
    
    public func action(forState state: Game.State) -> Game.Action {
        assert(!state.isTerminal)
        assert(!state.legalActions.isEmpty)
 
        let legalUCBValues = state.legalActions.map{ UCB[$0]! }
        let maxLegalUCBAction = state.legalActions[legalUCBValues.argMax()!]
        return maxLegalUCBAction
    }
    
    public mutating func update(with action: Game.Action, reward: Double) {
        N[action] = N[action]! + 1
        Q[action] = Q[action]! + (reward - Q[action]!) / Double(N[action]!)
    }
}



/// Upper Confidence Bound Action Selection algorithm per Chapter 2.7
public struct GradientGambler: StochasticPolicy {
    public typealias Game = MultiArmedBandit
    
    /// Instance of the Multi-armed Bandit game being played
    private let game: Game

    /// Value estimation update rate
    private var α: Double

    /// Number of times a state has been visited
    private var H: [Game.Action: Double]
    
    /// Reward baseline
    private var R: Double
    
    public init(_ game: Game, α: Double = 0.01) {
        self.game = game
        self.α = α

        let initialH = Tensor(randomUniform: TensorShape([game.allActions.count]),
                              lowerBound: Tensor(0.0),
                              upperBound: Tensor(1e-5)).scalars
        H = Dictionary(uniqueKeysWithValues: zip(game.allActions, initialH))
        
        R = 0.0
    }

    public func actionProbabilities(forState state: Game.State) -> [Game.Action : Double] {
        assert(!state.isTerminal)
        assert(!state.legalActions.isEmpty)

        let legalActionsExpLogits = state.legalActions.map { exp(H[$0]!) }
        let logitsExpSum = legalActionsExpLogits.reduce(0.0, +)
        let legalActionsProbabilities = legalActionsExpLogits.map { $0 / logitsExpSum }
        
        return Dictionary(uniqueKeysWithValues: zip(state.legalActions, legalActionsProbabilities))
    }
    
    public mutating func update(with takenAction: Game.Action, in state: Game.State, reward: Double) {
        let actionProbabilities = self.actionProbabilities(forState: state)

        for (action, h) in H {
            if action == takenAction {
                H[takenAction] = h + α * (reward - R) * (1.0 - actionProbabilities[takenAction]!)
            } else {
                H[action] = h + α * (reward - R) * actionProbabilities[action]!
            }
        }

        R = R + α * (reward - R)
    }
}
