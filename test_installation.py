#!/usr/bin/env python3
"""
Test script to verify installation and basic functionality
"""

import sys
import cv2
import numpy as np

def test_opencv():
    """Test OpenCV installation"""
    print("Testing OpenCV installation...")
    try:
        version = cv2.__version__
        print(f"✓ OpenCV version: {version}")
        
        # Test camera access
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera access successful")
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera frame captured: {frame.shape}")
            cap.release()
        else:
            print("✗ Camera access failed - check camera permissions")
            return False
            
        return True
    except Exception as e:
        print(f"✗ OpenCV test failed: {e}")
        return False

def test_numpy():
    """Test NumPy installation"""
    print("\nTesting NumPy installation...")
    try:
        version = np.__version__
        print(f"✓ NumPy version: {version}")
        
        # Test basic operations
        arr = np.array([1, 2, 3, 4, 5])
        mean = np.mean(arr)
        print(f"✓ NumPy operations working: mean([1,2,3,4,5]) = {mean}")
        
        return True
    except Exception as e:
        print(f"✗ NumPy test failed: {e}")
        return False

def test_modules():
    """Test custom modules"""
    print("\nTesting custom modules...")
    try:
        from card_detector import CardDetector
        print("✓ card_detector module imported")
        
        from strategy_engine import StrategyEngine
        print("✓ strategy_engine module imported")
        
        from ui_overlay import UIOverlay
        print("✓ ui_overlay module imported")
        
        # Test instantiation
        detector = CardDetector()
        print("✓ CardDetector instantiated")
        
        strategy = StrategyEngine()
        print("✓ StrategyEngine instantiated")
        
        ui = UIOverlay()
        print("✓ UIOverlay instantiated")
        
        return True
    except Exception as e:
        print(f"✗ Module test failed: {e}")
        print("  Make sure all .py files are in the same directory")
        return False

def test_display():
    """Test display functionality"""
    print("\nTesting display window...")
    try:
        # Create a test image
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_img, "Blackjack Analyzer Test", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(test_img, "Press any key to continue", (150, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        cv2.imshow("Test Window", test_img)
        print("✓ Display window created")
        print("  Press any key in the window to continue...")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("✓ Display test completed")
        
        return True
    except Exception as e:
        print(f"✗ Display test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Blackjack Analyzer Installation Test")
    print("=" * 50)
    
    tests = [
        test_opencv,
        test_numpy,
        test_modules,
        test_display
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)
    
    if all(results):
        print("✓ All tests passed! Installation successful.")
        print("\nYou can now run the main application with:")
        print("  python main.py")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())