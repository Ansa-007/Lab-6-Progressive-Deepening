"""
Comprehensive Test Suite for Progressive Deepening Implementation
Industry-Grade Testing with Validation and Benchmarking

This test suite provides:
- Unit tests for all core functions
- Integration tests for complex scenarios
- Performance benchmarks
- Edge case validation
- Industry use case testing
"""

import unittest
import time
import sys
import os
from typing import Dict, List, Any

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Progressive_Deepening_Lab_Manual import (
    validate_graph,
    depth_limited_dfs,
    iterative_deepening_dfs,
    progressive_deepening_with_heuristics,
    bidirectional_progressive_deepening,
    performance_analysis,
    create_industry_test_cases,
    ProgressiveDeepeningError,
    SearchResult
)


class TestProgressiveDeepening(unittest.TestCase):
    """Comprehensive test suite for Progressive Deepening algorithms."""
    
    def setUp(self):
        """Set up test fixtures for all test methods."""
        self.simple_graph = {
            'A': ['B', 'C'],
            'B': ['D', 'E'],
            'C': ['F'],
            'D': [], 'E': [], 'F': []
        }
        
        self.linear_graph = {
            '1': ['2'],
            '2': ['3'],
            '3': ['4'],
            '4': ['5'],
            '5': []
        }
        
        self.star_graph = {
            'Center': ['A', 'B', 'C', 'D', 'E'],
            'A': [], 'B': [], 'C': [], 'D': [], 'E': []
        }
        
        self.complex_graph = {
            'Start': ['A', 'B'],
            'A': ['C', 'D'],
            'B': ['E', 'F'],
            'C': ['G'],
            'D': ['H'],
            'E': ['I'],
            'F': ['J'],
            'G': [], 'H': [], 'I': [], 'J': []
        }
    
    def test_validate_graph(self):
        """Test graph validation functionality."""
        # Valid graphs
        self.assertIsNone(validate_graph(self.simple_graph))
        self.assertIsNone(validate_graph(self.linear_graph))
        self.assertIsNone(validate_graph(self.star_graph))
        
        # Invalid graphs
        with self.assertRaises(ProgressiveDeepeningError):
            validate_graph({})
        
        with self.assertRaises(ProgressiveDeepeningError):
            validate_graph({'A': 'not_a_list'})
        
        with self.assertRaises(ProgressiveDeepeningError):
            validate_graph({'A': ['B']})  # B not in graph keys
        
        with self.assertRaises(ProgressiveDeepeningError):
            validate_graph("not_a_dict")
    
    def test_depth_limited_dfs(self):
        """Test depth-limited DFS functionality."""
        # Test successful search within depth limit
        found, path, nodes = depth_limited_dfs(self.simple_graph, 'A', 'D', 3)
        self.assertTrue(found)
        self.assertIn('D', path)
        self.assertGreater(nodes, 0)
        
        # Test search beyond depth limit
        found, path, nodes = depth_limited_dfs(self.simple_graph, 'A', 'D', 1)
        self.assertFalse(found)
        self.assertEqual(path, [])
        
        # Test linear graph
        found, path, nodes = depth_limited_dfs(self.linear_graph, '1', '5', 4)
        self.assertTrue(found)
        self.assertEqual(path, ['1', '2', '3', '4', '5'])
        
        # Test zero depth limit
        found, path, nodes = depth_limited_dfs(self.simple_graph, 'A', 'A', 0)
        self.assertTrue(found)
        self.assertEqual(path, ['A'])
    
    def test_iterative_deepening_dfs(self):
        """Test Iterative Deepening DFS implementation."""
        # Test successful search
        result = iterative_deepening_dfs(self.simple_graph, 'A', 'F')
        self.assertIsInstance(result, SearchResult)
        self.assertTrue(result.found)
        self.assertIn('F', result.path)
        self.assertEqual(result.path[0], 'A')
        self.assertEqual(result.path[-1], 'F')
        self.assertGreater(result.time_elapsed, 0)
        self.assertGreaterEqual(result.max_depth, 0)
        
        # Test search for same node
        result = iterative_deepening_dfs(self.simple_graph, 'A', 'A')
        self.assertTrue(result.found)
        self.assertEqual(result.path, ['A'])
        self.assertEqual(result.max_depth, 0)
        
        # Test linear graph
        result = iterative_deepening_dfs(self.linear_graph, '1', '5')
        self.assertTrue(result.found)
        self.assertEqual(result.path, ['1', '2', '3', '4', '5'])
        
        # Test star graph
        result = iterative_deepening_dfs(self.star_graph, 'Center', 'E')
        self.assertTrue(result.found)
        self.assertEqual(result.path[0], 'Center')
        self.assertEqual(result.path[-1], 'E')
        
        # Test error cases
        with self.assertRaises(ProgressiveDeepeningError):
            iterative_deepening_dfs(self.simple_graph, 'X', 'A')
        
        with self.assertRaises(ProgressiveDeepeningError):
            iterative_deepening_dfs(self.simple_graph, 'A', 'X')
    
    def test_progressive_deepening_with_heuristics(self):
        """Test heuristic-guided progressive deepening."""
        # Test with custom heuristics
        heuristic = {'A': 0.1, 'B': 0.3, 'C': 0.2, 'D': 0.5, 'E': 0.4, 'F': 0.6}
        result = progressive_deepening_with_heuristics(
            self.simple_graph, 'A', 'F', heuristic
        )
        self.assertTrue(result.found)
        self.assertIn('F', result.path)
        
        # Test without heuristics (default)
        result = progressive_deepening_with_heuristics(self.simple_graph, 'A', 'F')
        self.assertTrue(result.found)
        
        # Test with max depth limit
        result = progressive_deepening_with_heuristics(
            self.simple_graph, 'A', 'F', max_depth=2
        )
        # Should still find F as it's within depth 2
        self.assertTrue(result.found)
    
    def test_bidirectional_progressive_deepening(self):
        """Test bidirectional progressive deepening."""
        # Test successful bidirectional search
        result = bidirectional_progressive_deepening(self.simple_graph, 'A', 'F')
        self.assertTrue(result.found)
        self.assertIn('F', result.path)
        
        # Test linear graph
        result = bidirectional_progressive_deepening(self.linear_graph, '1', '5')
        self.assertTrue(result.found)
        
        # Test star graph
        result = bidirectional_progressive_deepening(self.star_graph, 'Center', 'D')
        self.assertTrue(result.found)
        
        # Test with depth limit
        result = bidirectional_progressive_deepening(
            self.simple_graph, 'A', 'F', max_depth=3
        )
        self.assertTrue(result.found)
    
    def test_performance_analysis(self):
        """Test performance analysis functionality."""
        results = performance_analysis(self.simple_graph, 'A', 'F')
        
        # Check all strategies are present
        expected_strategies = ['iddfs', 'progressive_heuristic', 'bidirectional']
        for strategy in expected_strategies:
            self.assertIn(strategy, results)
            self.assertIsInstance(results[strategy], SearchResult)
        
        # All strategies should find the target
        for strategy, result in results.items():
            self.assertTrue(result.found, f"Strategy {strategy} should find target")
            self.assertGreater(result.time_elapsed, 0)
            self.assertGreaterEqual(result.nodes_explored, 0)
    
    def test_industry_test_cases(self):
        """Test industry-relevant test cases."""
        test_cases = create_industry_test_cases()
        
        # Check all test cases are present
        expected_cases = [
            'network_topology', 'software_dependencies', 
            'supply_chain', 'social_network'
        ]
        for case in expected_cases:
            self.assertIn(case, test_cases)
            self.assertIsInstance(test_cases[case], dict)
            self.assertGreater(len(test_cases[case]), 0)
        
        # Test network topology
        network_graph = test_cases['network_topology']
        result = iterative_deepening_dfs(network_graph, 'Core_Router', 'Server_8')
        self.assertTrue(result.found)
        
        # Test software dependencies
        dep_graph = test_cases['software_dependencies']
        result = iterative_deepening_dfs(dep_graph, 'main_app', 'style_framework')
        self.assertTrue(result.found)
        
        # Test supply chain
        supply_graph = test_cases['supply_chain']
        result = iterative_deepening_dfs(supply_graph, 'Manufacturer', 'Retailer_8')
        self.assertTrue(result.found)
        
        # Test social network
        social_graph = test_cases['social_network']
        result = iterative_deepening_dfs(social_graph, 'User_A', 'User_P')
        self.assertTrue(result.found)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Single node graph
        single_node = {'A': []}
        result = iterative_deepening_dfs(single_node, 'A', 'A')
        self.assertTrue(result.found)
        self.assertEqual(result.path, ['A'])
        
        # Two node graph
        two_nodes = {'A': ['B'], 'B': []}
        result = iterative_deepening_dfs(two_nodes, 'A', 'B')
        self.assertTrue(result.found)
        self.assertEqual(result.path, ['A', 'B'])
        
        # Disconnected graph (should fail)
        disconnected = {'A': ['B'], 'B': [], 'C': ['D'], 'D': []}
        result = iterative_deepening_dfs(disconnected, 'A', 'D')
        self.assertFalse(result.found)
        self.assertEqual(result.path, [])
    
    def test_performance_characteristics(self):
        """Test performance characteristics and scalability."""
        # Create progressively larger graphs
        sizes = [5, 10, 15]
        results = {}
        
        for size in sizes:
            # Create a linear graph of given size
            graph = {}
            for i in range(size):
                node = f"N{i}"
                if i < size - 1:
                    graph[node] = [f"N{i+1}"]
                else:
                    graph[node] = []
            
            start_time = time.perf_counter()
            result = iterative_deepening_dfs(graph, 'N0', f'N{size-1}')
            end_time = time.perf_counter()
            
            results[size] = {
                'time': result.time_elapsed,
                'nodes_explored': result.nodes_explored,
                'max_depth': result.max_depth
            }
            
            self.assertTrue(result.found)
            self.assertEqual(result.max_depth, size - 1)
        
        # Verify that larger graphs take more time (generally)
        self.assertGreater(results[15]['time'], results[5]['time'])
        self.assertGreater(results[15]['nodes_explored'], results[5]['nodes_explored'])
    
    def test_memory_efficiency(self):
        """Test memory efficiency characteristics."""
        # Create a graph with high branching factor
        high_branching = {
            'Root': [f'Child_{i}' for i in range(10)]
        }
        for i in range(10):
            high_branching[f'Child_{i}'] = []
        
        # The algorithm should handle this without excessive memory usage
        result = iterative_deepening_dfs(high_branching, 'Root', 'Child_9')
        self.assertTrue(result.found)
        
        # Max depth should be 1 (root -> child)
        self.assertEqual(result.max_depth, 1)


class TestBenchmarking(unittest.TestCase):
    """Benchmarking tests for performance analysis."""
    
    def setUp(self):
        """Set up benchmarking fixtures."""
        self.test_cases = create_industry_test_cases()
    
    def test_algorithm_comparison(self):
        """Compare performance across different algorithms."""
        for case_name, graph in self.test_cases.items():
            with self.subTest(case=case_name):
                nodes = list(graph.keys())
                start, target = nodes[0], nodes[-1]
                
                results = performance_analysis(graph, start, target)
                
                # All strategies should complete
                for strategy, result in results.items():
                    self.assertIsInstance(result, SearchResult)
                    self.assertGreaterEqual(result.time_elapsed, 0)
                    self.assertGreaterEqual(result.nodes_explored, 0)
                
                # At least one strategy should find the target
                found_strategies = [s for s, r in results.items() if r.found]
                self.assertGreater(len(found_strategies), 0, 
                                 f"No strategy found target in {case_name}")
    
    def test_scalability_analysis(self):
        """Test scalability characteristics."""
        # Test graphs of different sizes
        sizes = [5, 10, 20]
        
        for size in sizes:
            with self.subTest(size=size):
                # Create a balanced tree
                graph = {}
                for i in range(size):
                    children = []
                    left_child = 2 * i + 1
                    right_child = 2 * i + 2
                    
                    if left_child < size:
                        children.append(f"N{left_child}")
                    if right_child < size:
                        children.append(f"N{right_child}")
                    
                    graph[f"N{i}"] = children
                
                # Test performance
                start_time = time.perf_counter()
                result = iterative_deepening_dfs(graph, 'N0', f'N{size-1}')
                end_time = time.perf_counter()
                
                # Should complete within reasonable time
                self.assertLess(result.time_elapsed, 1.0, 
                              f"Graph size {size} took too long")


def run_comprehensive_tests():
    """Run all tests and provide summary."""
    print("=== Progressive Deepening Comprehensive Test Suite ===\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestProgressiveDeepening))
    suite.addTests(loader.loadTestsFromTestCase(TestBenchmarking))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n=== Test Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Set recursion limit for deep graphs
    sys.setrecursionlimit(10000)
    
    # Run comprehensive tests
    success = run_comprehensive_tests()
    
    if success:
        print("\n[SUCCESS] All tests passed! Progressive Deepening implementation is working correctly.")
    else:
        print("\n[FAILED] Some tests failed. Please review the implementation.")
    
    sys.exit(0 if success else 1)
