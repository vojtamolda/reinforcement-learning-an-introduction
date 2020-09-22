import TensorFlow
import OpenSpiel
import XCTest

import MultiArmedBandit
import Utilities


fileprivate func play<Policy: StochasticPolicy>(_ game: Policy.Game, actingWith policy: Policy,
                                                for steps: Int) -> Policy.Game.State {
    var currentState = game.initialState
    
    for _ in 0..<steps {
        assert(!currentState.legalActions.isEmpty)
        
        let actionProbabilities = policy.actionProbabilities(forState: currentState)
        assert(!actionProbabilities.isEmpty)

        let sampledAction = actionProbabilities.sample()!
        currentState = currentState.applying(sampledAction)
    }
    
    return currentState
}



final class GamblerTests: XCTestCase {

    

    
    func testGame() {
        let bandit = MultiArmedBandit()

        let policies: [AnyStochasticPolicy<MultiArmedBandit>] = [
            .init(UniformRandomPolicy(bandit)),
            .init(EpsilonGreedyGambler(bandit)),
            .init(AveragingGambler(bandit)),
            .init(OptimisticGambler(bandit)),
            .init(UCBGambler(bandit)),
            .init(GradientGambler(bandit))
        ]
        
        for policy in policies {
            let finalState = play(bandit, actingWith: policy, for: 100)
        }
    }
    
    
    func testRandomActions() {
        let bandit = MultiArmedBandit(armCount: 10)
        print(bandit)
    }
    
    func testRandomSeedRepeatability() {
        let (bandit0, bandit1) = (MultiArmedBandit(), MultiArmedBandit())
        
        print(bandit0)
        print(bandit1)
        
//        let seed = TensorFlow.randomSeedForTensorFlow()
//        let rewards = TensorFlow.withRandomSeedForTensorFlow(seed) { _ -> [Double] in
//            let agent = UniformRandomPolicy(bandit0)
//        }
        
    }
}
