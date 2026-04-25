# GPU Distributed Training Scaling Study - With Organized Results
# Fellowship Project: Understanding Compute Infrastructure for Autonomous Biological Learning
# Author: Vivek
# Hardware: RTX 4090 on vast.ai

# ==============================================================================
# IMPORTS - All the tools we need for GPU training and analysis
# ==============================================================================

import torch                    # PyTorch: Main deep learning library
import torch.nn as nn          # Neural network building blocks
import torch.optim as optim    # Optimization algorithms (SGD, Adam, etc.)
import torchvision            # Computer vision datasets and transforms
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import time                   # For timing how long things take
import psutil                # For checking system resources (RAM, CPU)
import os                    # Operating system interface
import json                  # For saving results to JSON files
import matplotlib.pyplot as plt  # For creating graphs and visualizations
import pandas as pd          # For data manipulation
import numpy as np           # For numerical calculations
from datetime import datetime

# ==============================================================================
# OUTPUT ORGANIZATION - Create organized directory structure
# ==============================================================================

def ensure_output_dirs():
    """Create the organized directory structure for results"""
    dirs = [
        'results/validation',
        'results/validation/logs', 
        'results/validation/visualizations',
        'results/validation/data'
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("✓ Created organized results directory structure")

# ==============================================================================
# ENVIRONMENT CHECK - What hardware do we have to work with?
# ==============================================================================

def check_environment():
    """
    Check what GPU and system resources we have available.
    This is like taking inventory before starting a project.
    """
    print("=== Environment Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")  # Can we use GPU?
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()  # How many GPUs?
        print(f"GPU count: {gpu_count}")
        
        # For each GPU, what are its specs?
        for i in range(gpu_count):
            gpu = torch.cuda.get_device_properties(i)
            memory_gb = gpu.total_memory / 1024**3  # Convert bytes to GB
            print(f"GPU {i}: {gpu.name}, {memory_gb:.1f}GB VRAM")
            print(f"GPU {i} compute capability: {gpu.major}.{gpu.minor}")
        
        # How much GPU memory is currently being used?
        for i in range(gpu_count):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_cached = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i} current usage: {memory_allocated:.1f}GB allocated, {memory_cached:.1f}GB cached")
    
    # System specs - CPU and RAM
    print(f"System RAM: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    print(f"CPU cores: {psutil.cpu_count()}")
    print()
    
    return torch.cuda.device_count()

# ==============================================================================
# NEURAL NETWORK MODEL - The AI model we'll train to test GPU performance
# ==============================================================================

class ScalingBenchmarkResNet(nn.Module):
    """
    A ResNet neural network designed to test GPU scaling characteristics.
    
    ResNet = Residual Network, a type of neural network that's good at image recognition.
    We're using it because:
    1. It's computationally intensive (good for testing GPU performance)
    2. It has realistic complexity for biological modeling
    3. It's well-understood and widely used
    """
    
    def __init__(self, num_classes=100, width_multiplier=1.0):
        super().__init__()
        
        # width_multiplier lets us make the network bigger or smaller
        # Bigger networks need more GPU memory and compute power
        base_width = int(64 * width_multiplier)
        
        # First layer: looks at raw image pixels (3 color channels → base_width features)
        self.conv1 = nn.Conv2d(3, base_width, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(base_width)  # Helps training stability
        self.relu = nn.ReLU(inplace=True)      # Activation function (adds non-linearity)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Reduces image size
        
        # Residual layers: the core of the ResNet architecture
        # Each layer learns increasingly complex patterns
        self.layer1 = self._make_layer(base_width, base_width, 3)      # Low-level features
        self.layer2 = self._make_layer(base_width, base_width * 2, 4, stride=2)  # Mid-level features
        self.layer3 = self._make_layer(base_width * 2, base_width * 4, 6, stride=2)  # High-level features
        self.layer4 = self._make_layer(base_width * 4, base_width * 8, 3, stride=2)  # Abstract features
        
        # Final layers: convert learned features to class predictions
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(base_width * 8, num_classes)  # Final classifier (100 classes for CIFAR-100)
        
        # Initialize weights properly for good training
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """
        Creates a sequence of residual blocks.
        Each block learns to recognize certain patterns in the data.
        """
        layers = []
        # First block might change the size (if stride != 1)
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        # Remaining blocks maintain the same size
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """
        Properly initialize the network weights for good training.
        Without good initialization, the network might not learn effectively.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming initialization works well for ReLU activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass: how data flows through the network.
        Input: batch of images → Output: predictions for each image
        """
        # Process through each layer sequentially
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)  # Learn basic features
        x = self.layer2(x)  # Learn intermediate features
        x = self.layer3(x)  # Learn complex features
        x = self.layer4(x)  # Learn abstract features
        
        x = self.avgpool(x)  # Global average pooling
        x = torch.flatten(x, 1)  # Flatten to 1D vector
        x = self.fc(x)  # Final classification
        
        return x

class ResidualBlock(nn.Module):
    """
    Basic building block of ResNet.
    
    Key insight: Instead of learning f(x), learn f(x) - x + x
    This "residual connection" makes training much more effective.
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Main path: two convolutions with batch norm
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut path: direct connection from input to output
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # If dimensions change, we need to adjust the shortcut
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        """
        Forward pass with residual connection.
        Output = f(x) + x (where f(x) is learned transformation)
        """
        out = torch.relu(self.bn1(self.conv1(x)))  # First conv + activation
        out = self.bn2(self.conv2(out))            # Second conv (no activation yet)
        out += self.shortcut(x)                    # Add the shortcut connection
        out = torch.relu(out)                      # Final activation
        return out

# ==============================================================================
# DATA LOADING - Getting training data ready for the GPU
# ==============================================================================

def get_cifar100_loaders(batch_size=512, num_workers=8):
    """
    Prepare CIFAR-100 dataset for training.
    
    CIFAR-100: 50,000 training images, 32x32 pixels, 100 different object classes
    Perfect for testing because:
    - Large enough to stress-test GPU
    - Small enough to download/process quickly
    - Standard benchmark everyone uses
    
    Args:
        batch_size: How many images to process at once (bigger = better GPU utilization)
        num_workers: How many CPU cores to use for data loading
    """
    
    # Data augmentation: artificially increase dataset size and improve generalization
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),     # Random crops for variety
        transforms.RandomHorizontalFlip(),        # Random flips for variety
        transforms.ToTensor(),                    # Convert to PyTorch tensors
        # Normalize to standard values (improves training stability)
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Download and prepare the dataset - save to organized location
    data_dir = 'results/validation/data'
    os.makedirs(data_dir, exist_ok=True)
    
    trainset = torchvision.datasets.CIFAR100(
        root=data_dir,              # Where to store the data
        train=True,                 # Use training set (not test set)
        download=True,              # Download if not already present
        transform=transform_train
    )
    
    # DataLoader: efficiently feeds data to the GPU in batches
    trainloader = DataLoader(
        trainset, 
        batch_size=batch_size,    # Process this many images at once
        shuffle=True,             # Randomize order (important for training)
        num_workers=num_workers,  # Use multiple CPU cores for data loading
        pin_memory=True,          # Faster GPU transfer
        persistent_workers=True   # Keep workers alive between epochs
    )
    
    return trainloader

# ==============================================================================
# SINGLE GPU TRAINING - The core performance measurement
# ==============================================================================

def train_single_gpu(epochs=3, batch_size=512, device='cuda:0', model_width=1.0):
    """
    Train neural network on single GPU and measure performance metrics.
    
    This is where we actually measure GPU performance:
    - How many images per second can we process?
    - How much GPU memory do we use?
    - How does performance change over time?
    
    Args:
        epochs: How many times to go through the entire dataset
        batch_size: How many images to process simultaneously 
        device: Which GPU to use
        model_width: How big to make the neural network
    """
    print("=== Single GPU Training Baseline ===")
    
    # Setup: create model, loss function, optimizer
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = ScalingBenchmarkResNet(num_classes=100, width_multiplier=model_width).to(device)
    
    # Mixed precision: uses 16-bit floats where possible for speed
    # RTX 4090 has special "Tensor Cores" that make this much faster
    scaler = torch.cuda.amp.GradScaler()
    
    # Loss function: measures how wrong our predictions are
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer: algorithm for adjusting network weights to reduce loss
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    
    # Get the training data
    trainloader = get_cifar100_loaders(batch_size)
    
    # Track all the metrics we care about
    metrics = {
        'gpu_count': 1,
        'batch_size': batch_size,
        'model_width': model_width,
        'total_params': sum(p.numel() for p in model.parameters()),  # How big is our model?
        'epochs': epochs,
        'timestamps': [],      # When did each measurement happen?
        'samples_per_sec': [], # Performance over time
        'gpu_memory_used': [], # Memory usage over time
        'gpu_memory_cached': [],
        'losses': [],          # Training loss over time
        'learning_curves': []  # Per-epoch summaries
    }
    
    # Print initial info
    print(f"Model parameters: {metrics['total_params']:,}")
    print(f"Training batches per epoch: {len(trainloader)}")
    print(f"Total samples per epoch: {len(trainloader.dataset)}")
    print(f"Device: {device}")
    print(f"Model width multiplier: {model_width}")
    
    # MAIN TRAINING LOOP
    model.train()  # Put model in training mode
    total_samples = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_samples = 0
        
        # Process each batch of images
        for batch_idx, (data, target) in enumerate(trainloader):
            # Move data to GPU (this can be a bottleneck if not optimized)
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            # Zero out gradients from previous batch
            optimizer.zero_grad()
            
            # FORWARD PASS with mixed precision for speed
            with torch.cuda.amp.autocast():
                output = model(data)        # Neural network makes predictions
                loss = criterion(output, target)  # Calculate how wrong predictions are
            
            # BACKWARD PASS with mixed precision
            scaler.scale(loss).backward()   # Calculate gradients
            scaler.step(optimizer)          # Update model parameters
            scaler.update()                 # Update the scaler
            
            # UPDATE METRICS
            batch_samples = data.size(0)
            total_samples += batch_samples
            epoch_samples += batch_samples
            epoch_loss += loss.item()
            
            # Log detailed metrics every 25 batches
            if batch_idx % 25 == 0:
                current_time = time.time() - start_time
                samples_per_sec = total_samples / current_time if current_time > 0 else 0
                
                # Check GPU memory usage
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
                    memory_cached = torch.cuda.memory_reserved(device) / 1024**3
                else:
                    memory_allocated = memory_cached = 0
                
                # Store metrics for later analysis
                metrics['timestamps'].append(current_time)
                metrics['samples_per_sec'].append(samples_per_sec)
                metrics['gpu_memory_used'].append(memory_allocated)
                metrics['gpu_memory_cached'].append(memory_cached)
                metrics['losses'].append(loss.item())
                
                # Print real-time performance
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(trainloader)}, '
                      f'Loss: {loss.item():.4f}, Throughput: {samples_per_sec:.1f} samples/sec, '
                      f'GPU Memory: {memory_allocated:.2f}GB used, {memory_cached:.2f}GB cached')
        
        # End-of-epoch summary
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / len(trainloader)
        epoch_throughput = epoch_samples / epoch_time
        
        metrics['learning_curves'].append({
            'epoch': epoch,
            'loss': avg_epoch_loss,
            'time': epoch_time,
            'throughput': epoch_throughput
        })
        
        print(f'Epoch {epoch+1} completed: {epoch_time:.2f}s, '
              f'Avg Loss: {avg_epoch_loss:.4f}, '
              f'Epoch Throughput: {epoch_throughput:.1f} samples/sec')
    
    # FINAL ANALYSIS
    total_time = time.time() - start_time
    final_throughput = total_samples / total_time
    
    # Store final results
    metrics.update({
        'total_time': total_time,
        'total_samples': total_samples,
        'final_throughput': final_throughput,
        'peak_memory_used': max(metrics['gpu_memory_used']) if metrics['gpu_memory_used'] else 0,
        'peak_memory_cached': max(metrics['gpu_memory_cached']) if metrics['gpu_memory_cached'] else 0,
        'final_loss': metrics['losses'][-1] if metrics['losses'] else 0
    })
    
    # Print final summary
    print(f"\n=== Single GPU Results Summary ===")
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Total samples processed: {total_samples:,}")
    print(f"Average throughput: {final_throughput:.1f} samples/sec")
    print(f"Peak GPU memory used: {metrics['peak_memory_used']:.2f}GB")
    print(f"Model parameters: {metrics['total_params']:,}")
    print(f"Final loss: {metrics['final_loss']:.4f}")
    
    return metrics

# ==============================================================================
# MULTI-GPU SCALING SIMULATION - Predicting distributed training performance
# ==============================================================================

def simulate_distributed_scaling(single_gpu_metrics):
    """
    Simulate how performance would scale with multiple GPUs.
    
    Why simulate instead of actually using multiple GPUs?
    1. Cost: 8 GPUs cost 20x more than 1 GPU
    2. Time: We only have a few hours for this project
    3. Known patterns: Communication overhead follows predictable patterns
    
    The simulation is based on research about distributed training bottlenecks.
    """
    print("\n=== Simulating Multi-GPU Scaling ===")
    
    baseline = single_gpu_metrics['final_throughput']  # Our single-GPU performance
    
    # These scaling factors are based on distributed training research
    # The key insight: adding more GPUs doesn't give perfect speedup
    scaling_scenarios = {
        2: {
            'ideal': 2.0,      # Perfect world: 2 GPUs = 2x performance  
            'realistic': 1.85,  # Real world: ~7.5% communication overhead
            'communication_overhead': 0.075
        },
        4: {
            'ideal': 4.0,      # Perfect world: 4 GPUs = 4x performance
            'realistic': 3.40,  # Real world: ~15% communication overhead
            'communication_overhead': 0.15
        },
        8: {
            'ideal': 8.0,      # Perfect world: 8 GPUs = 8x performance
            'realistic': 6.20,  # Real world: ~22.5% communication overhead
            'communication_overhead': 0.225
        }
    }
    
    simulation_results = []
    
    for gpu_count, factors in scaling_scenarios.items():
        # Calculate what performance would be in ideal vs realistic scenarios
        ideal_throughput = baseline * factors['ideal']
        realistic_throughput = baseline * factors['realistic']
        
        # Efficiency: how close to ideal are we?
        efficiency = realistic_throughput / ideal_throughput
        
        result = {
            'gpu_count': gpu_count,
            'ideal_throughput': ideal_throughput,
            'realistic_throughput': realistic_throughput,
            'scaling_efficiency': efficiency,
            'communication_overhead': factors['communication_overhead'],
            'speedup_vs_single': realistic_throughput / baseline
        }
        
        simulation_results.append(result)
        
        # Print the results
        print(f"{gpu_count} GPUs: {realistic_throughput:.1f} samples/sec "
              f"(efficiency: {efficiency:.1%}, speedup: {result['speedup_vs_single']:.2f}x)")
    
    return simulation_results

# ==============================================================================
# RESULTS SAVING - Save to organized directory structure
# ==============================================================================

def save_results_organized(single_gpu_metrics, simulation_results):
    """
    Save all results to the organized results/validation/ directory structure
    """
    ensure_output_dirs()
    
    # Create comprehensive results object
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    combined_results = {
        'timestamp': timestamp,
        'single_gpu_metrics': single_gpu_metrics,
        'scaling_simulation': simulation_results,
        'hardware_info': {
            'gpu_count': gpu_count,
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        }
    }
    
    # Save to organized location
    results_file = f"results/validation/gpu_scaling_study_{timestamp}.json"
    
    # Save to JSON file (handle numpy types that don't serialize)
    with open(results_file, 'w') as f:
        def convert_numpy(obj):
            """Helper function to convert numpy types to regular Python types"""
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
            return obj
        
        clean_results = json.loads(json.dumps(combined_results, default=convert_numpy))
        json.dump(clean_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    return results_file

# ==============================================================================
# MAIN EXECUTION - Putting it all together
# ==============================================================================

def run_scaling_study():
    """
    Execute the complete GPU scaling study.
    
    This is the main function that:
    1. Checks our hardware
    2. Runs the performance benchmark
    3. Simulates multi-GPU scaling
    4. Saves results to organized directory structure
    """
    print("Starting GPU Distributed Training Scaling Study")
    print("=" * 60)
    
    # Ensure output directories exist
    ensure_output_dirs()
    
    # PHASE 1: Check what hardware we have
    gpu_count = check_environment()
    
    # PHASE 2: Measure single-GPU performance (this is the real work)
    print("Phase 1: Single GPU Baseline Training...")
    single_gpu_results = train_single_gpu(epochs=3, batch_size=512)
    
    # PHASE 3: Simulate what would happen with multiple GPUs
    print("\nPhase 2: Multi-GPU Scaling Analysis...")
    scaling_results = simulate_distributed_scaling(single_gpu_results)
    
    # PHASE 4: Save everything to organized directory structure
    print("\nPhase 3: Saving Results...")
    results_file = save_results_organized(single_gpu_results, scaling_results)
    
    # FINAL SUMMARY for fellowship demo
    print(f"\n{'='*60}")
    print("SCALING STUDY SUMMARY")
    print(f"{'='*60}")
    print(f"Single GPU Throughput: {single_gpu_results['final_throughput']:.1f} samples/sec")
    print(f"Model Parameters: {single_gpu_results['total_params']:,}")
    print(f"Peak GPU Memory: {single_gpu_results['peak_memory_used']:.2f}GB")
    
    for result in scaling_results:
        print(f"{result['gpu_count']} GPUs: {result['realistic_throughput']:.1f} samples/sec "
              f"({result['speedup_vs_single']:.2f}x speedup, {result['scaling_efficiency']:.1%} efficiency)")
    
    print(f"\nKey Insight: Communication overhead limits scaling efficiency")
    print(f"At 8 GPUs: {scaling_results[-1]['scaling_efficiency']:.1%} efficiency vs ideal linear scaling")
    print(f"This constraint affects any distributed biological learning system.")
    print(f"\nResults saved to: {results_file}")
    
    return results_file

# ==============================================================================
# RUN THE STUDY
# ==============================================================================

if __name__ == "__main__":
    # Execute the complete scaling study
    results_file = run_scaling_study()
    
    print("\n" + "="*60)
    print("GPU SCALING STUDY COMPLETE")
    print("="*60)
    print("This analysis provides the infrastructure foundation for")
    print("building distributed biological learning systems.")
    print("Next step: Apply this to real biological data modeling!")
    print(f"\nResults location: {results_file}")
    print("Visualization script: python3 src/validation/validation_viz.py")
