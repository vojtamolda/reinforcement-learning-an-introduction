import XCTest
import TensorFlow
import OpenSpiel
@testable import MultiArmedBandit


final class MultiArmedBanditTests: XCTestCase {

    lazy var bandits: [MultiArmedBandit] = {
        let armCount = Int.random(in: 1...100)
        return [
            MultiArmedBandit(stationary: true, armCount: armCount),
            MultiArmedBandit(stationary: false, armCount: armCount)
        ]
    }()
    
    func testSimpleGame() {
        for bandit in bandits {
            var state = bandit.initialState
            let initialRewards = state.armRewards
            XCTAssertEqual(initialRewards.scalarCount, bandit.armCount)

            var actions = [MultiArmedBandit.Action]()
            for _ in 0...1_000 {
                let action = state.legalActions.randomElement()!
                actions.append(action)
                state = state.applying(action)
            }
            XCTAssertEqual(actions, state.history)
            
            if bandit.stationary {
                XCTAssertEqual(initialRewards, state.armRewards)
            } else {
                XCTAssertNotEqual(initialRewards, state.armRewards)
            }
        }
    }
    
    func testRandomSeedRepeatability() {
        for bandit in bandits {
            let fixedActions = (0...1_000).map { _ in bandit.allActions.randomElement()! }
            let fixedSeed = randomSeedForTensorFlow()

            var state0 = bandit.initialState
            var utilities0 = [Double]()
            withRandomSeedForTensorFlow(fixedSeed) {
                state0 = bandit.initialState

                for fixedAction in fixedActions {
                    let utility = state0.utility(for: .player(0))
                    utilities0.append(utility)
                    
                    state0 = state0.applying(fixedAction)
                }
            }
            
            var state1 = bandit.initialState
            var utilities1 = [Double]()
            withRandomSeedForTensorFlow(fixedSeed) {
                state1 = bandit.initialState

                for fixedAction in fixedActions {
                    let utility = state1.utility(for: .player(0))
                    utilities1.append(utility)
                    
                    state1 = state1.applying(fixedAction)
                }
            }
            
            XCTAssertEqual(utilities0, utilities1)
            XCTAssertEqual(state0.armRewards, state1.armRewards)
            XCTAssertEqual(state0, state1)
        }
    }
}
