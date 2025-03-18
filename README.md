## Lab4: Cat & Dog Report

![Dr. Baek, cat whisperer](20250312_111537.jpg "Dr. Baek, cat whisperer")

### LiteRT model performance

The LiteRT model performed well on the Raspverry Pi 5 from a user experience perspective. It printed results many times a second and there was next to no noticable latency in the result updating. Because of its binary nature, it classified everything as a cat or a dog, outputting a floating point number where, roughly, positive values were more dog-like and negative values corresponded to more cat-like inputs. In theory, we could have added more thresholds such that, for instance, outputs with absolute values below 5.00 would be classified as "neither," but for the sake of simplicity, we omitted that feature. In practice, the classifier would reliably print "Dog" when the camera was pointed at a dog and "Cat" when pointed at a cat.

### Performance compared to Keras

The corresponding Keras model was quantized with a wider data type, and while it theoretically has better precision, in practice that is of little use in binary/trinary classification. Numerically, the Keras model took 8x as long to run (load, allocate, and execute one iteration).

The causes are more apparent when one looks at specific metric. Because the Keras model is simply *larger*, it does not fit into the Pi's rather small cache, meaning the CPU must make more requests to main memory. The Keras model entailed 30x more cache misses, 27x more memory accesses, and 31x more accesses which missed through all three levels of cache, thus causing the worst possible access penalty. The constant fetching from further and further memories, in turn, resulted in 13.8x more CPU stalls, and combined with Keras's more demanding numerical operations, cost 21x more CPU cycles total.

In short, the Keras model required more computation for useless precision and much, much more in the way of expensive memory accesses owing to its inability to be loaded entirely into cache.

### Data table

| Stat                   | Keras       | LiteRT     | Diff        | Factor      |
| ---------------------- | ----------- | ---------- | ----------- | ----------- |
| duration_time (ns)     | 13009878540 | 1576031265 | 11433847275 | 8.254835313 |
| cpu_cycles             | 31625004024 | 1496717095 | 30128286929 | 21.12958029 |
| cache-misses           | 253354708   | 8404318    | 244950390   | 30.1457784  |
| stalled-cycles-backend | 11242520473 | 814494145  | 10428026328 | 13.80307095 |
| L1-dcache-load-misses  | 254041726   | 8788863    | 245252863   | 28.90495915 |
| l2d_cache_inval        | 35945020    | 513631     | 35431389    | 69.98218565 |
| LLC-load-misses        | 174554132   | 5598791    | 168955341   | 31.17711163 |
| mem_access_rd          | 8500914063  | 311240116  | 8189673947  | 27.31304104 |

### Documentation

No help received aside from referencing TensorFlow documentation.