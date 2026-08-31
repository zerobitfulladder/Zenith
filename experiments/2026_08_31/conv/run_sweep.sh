set -e
P=/home/lavender/Projects/Zenith/.venv/bin/python
cd /home/lavender/Projects/Zenith/experiments/2026_08_31/conv
echo "########## MNIST ##########";        $P -u conv1.py mnist
echo "########## FASHION ##########";      $P -u conv1.py fashion_mnist
