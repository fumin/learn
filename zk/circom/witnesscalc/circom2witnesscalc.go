package witnesscalc

import (
	"fmt"
	"math/big"
	"sync"

	"github.com/pkg/errors"
	"github.com/wasmerio/wasmer-go/wasmer"
)

// Circom2WitnessCalculator is the object that allows performing witness calculation
// from signal inputs using the WitnessCalc WASM module.
type Circom2WitnessCalculator struct {
	instance            *wasmer.Instance
	sanityCheck         bool
	n32                 int32
	version             int32
	witnessSize         int32
	init                wasmer.NativeFunction
	getFieldNumLen32    wasmer.NativeFunction
	getInputSignalSize  wasmer.NativeFunction
	getInputSize        wasmer.NativeFunction
	getRawPrime         wasmer.NativeFunction
	getVersion          wasmer.NativeFunction
	getWitness          wasmer.NativeFunction
	readSharedRWMemory  wasmer.NativeFunction
	setInputSignal      wasmer.NativeFunction
	writeSharedRWMemory wasmer.NativeFunction

	errMsg *concurrentString
}

// NewCircom2WitnessCalculator creates a new WitnessCalculator from the WitnessCalc
// loaded WASM module in the runtime.
func NewCircom2WitnessCalculator(wasmBytes []byte, sanityCheck bool) (*Circom2WitnessCalculator, error) {
	engine := wasmer.NewEngine()
	store := wasmer.NewStore(engine)

	// Compiles the module
	module, _ := wasmer.NewModule(store, wasmBytes)

	limits, err := wasmer.NewLimits(2000, 100000)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	memType := wasmer.NewMemoryType(limits)

	memory := wasmer.NewMemory(store, memType)

	// Instantiates the module
	importObject := wasmer.NewImportObject()

	importObject.Register("env", map[string]wasmer.IntoExtern{
		"memory": memory,
	})

	errMsg := &concurrentString{}
	importObject.Register("runtime", map[string]wasmer.IntoExtern{
		"exceptionHandler":   getExceptionHandler(store, errMsg),
		"showSharedRWMemory": getShowSharedRWMemory(store),
		"printErrorMessage":  getLog(store),
		"writeBufferMessage": getLog(store),
	})

	instance, err := wasmer.NewInstance(module, importObject)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	// Gets the `init` exported function from the WebAssembly instance.
	init, err := instance.Exports.GetFunction("init")
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	// Calls that exported function with Go standard values. The WebAssembly
	// types are inferred and values are casted automatically.
	_, err = init(1)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	getFieldNumLen32, err := instance.Exports.GetFunction("getFieldNumLen32")
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	n32, err := getFieldNumLen32()
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	// this function is missing in wasm files generated with circom version prior to v2.0.4
	getInputSignalSize, _ := instance.Exports.GetFunction("getInputSignalSize")

	getInputSize, err := instance.Exports.GetFunction("getInputSize")
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	getRawPrime, err := instance.Exports.GetFunction("getRawPrime")
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	getVersion, err := instance.Exports.GetFunction("getVersion")
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	version, err := getVersion()
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	getWitness, err := instance.Exports.GetFunction("getWitness")
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	getWitnessSize, err := instance.Exports.GetFunction("getWitnessSize")
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	witnessSize, err := getWitnessSize()
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	setInputSignal, err := instance.Exports.GetFunction("setInputSignal")
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	readSharedRWMemory, err := instance.Exports.GetFunction("readSharedRWMemory")
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	writeSharedRWMemory, err := instance.Exports.GetFunction("writeSharedRWMemory")
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	return &Circom2WitnessCalculator{
		instance:            instance,
		sanityCheck:         sanityCheck,
		n32:                 n32.(int32),
		version:             version.(int32),
		witnessSize:         witnessSize.(int32),
		init:                init,
		getFieldNumLen32:    getFieldNumLen32,
		getInputSignalSize:  getInputSignalSize,
		getInputSize:        getInputSize,
		getRawPrime:         getRawPrime,
		getWitness:          getWitness,
		getVersion:          getVersion,
		setInputSignal:      setInputSignal,
		readSharedRWMemory:  readSharedRWMemory,
		writeSharedRWMemory: writeSharedRWMemory,
		errMsg:              errMsg,
	}, nil
}

// CalculateWitness calculates the witness given the inputs.
func (wc *Circom2WitnessCalculator) CalculateWitness(inputs map[string]any, sanityCheck bool) ([]*big.Int, error) {

	w := make([]*big.Int, wc.witnessSize)

	err := wc.doCalculateWitness(inputs, sanityCheck)
	if err != nil {
		return nil, err
	}

	for i := 0; i < int(wc.witnessSize); i++ {
		_, err := wc.getWitness(i)
		if err != nil {
			return nil, err
		}
		arr := make([]uint32, wc.n32)
		for j := 0; j < int(wc.n32); j++ {
			val, err := wc.readSharedRWMemory(int32(j))
			if err != nil {
				return nil, err
			}
			arr[int(wc.n32)-1-j] = uint32(val.(int32))
		}
		w[i] = fromArray32(arr)
	}

	errMsg := wc.errMsg.Get()
	if errMsg != "" {
		return nil, errors.Errorf("%s", errMsg)
	}

	return w, nil
}

// CalculateWitness calculates the witness given the inputs.
func (wc *Circom2WitnessCalculator) doCalculateWitness(inputs map[string]any, sanityCheck bool) error {
	//input is assumed to be a map from signals to arrays of bigInts
	sanityCheckVal := int32(0)
	if sanityCheck {
		sanityCheckVal = 1
	}
	_, err := wc.init(sanityCheckVal)
	if err != nil {
		return err
	}

	inputCounter := 0
	for inputName, inputValue := range inputs {
		hMSB, hLSB := fnvHash(inputName)
		fSlice := flatSlice(inputValue)

		if wc.getInputSignalSize != nil {
			signalSize, err := wc.getInputSignalSize(hMSB, hLSB)
			if err != nil {
				return err
			}

			if signalSize.(int32) < 0 {
				return fmt.Errorf("signal %s not found", inputName)
			}
			if len(fSlice) < int(signalSize.(int32)) {
				return fmt.Errorf("not enough values for input signal %s", inputName)
			}
			if len(fSlice) > int(signalSize.(int32)) {
				return fmt.Errorf("too many values for input signal %s", inputName)
			}
		}

		for i := range fSlice {
			arrFr, err := toArray32(fSlice[i], int(wc.n32))
			if err != nil {
				return err
			}
			for j := 0; j < int(wc.n32); j++ {
				_, err := wc.writeSharedRWMemory(j, int32(arrFr[int(wc.n32)-1-j]))
				if err != nil {
					return err
				}
			}
			_, err = wc.setInputSignal(hMSB, hLSB, i)
			if err != nil {
				return err
			}
			inputCounter++
		}
	}
	inputSize, err := wc.getInputSize()
	if inputCounter < int(inputSize.(int32)) {
		return fmt.Errorf("not all inputs have been set: only %d out of %d", inputCounter, inputSize)
	}
	return nil
}

func getExceptionHandler(store *wasmer.Store, ccStr *concurrentString) wasmer.IntoExtern {
	function := wasmer.NewFunction(
		store,
		wasmer.NewFunctionType(
			wasmer.NewValueTypes(wasmer.I32), // one i32 argument
			wasmer.NewValueTypes(),           // zero results
		),
		func(args []wasmer.Value) ([]wasmer.Value, error) {
			if len(args) > 0 {
				code := args[0].I32()
				var errStr string
				if code == 1 {
					errStr = "Signal not found. "
				} else if code == 2 {
					errStr = "Too many signals set. "
				} else if code == 3 {
					errStr = "Signal already set. "
				} else if code == 4 {
					errStr = "Assert Failed. "
				} else if code == 5 {
					errStr = "Not enough memory. "
				} else if code == 6 {
					errStr = "Input signal array access exceeds the size"
				} else {
					errStr = "Unknown error"
				}
				ccStr.Set("getExceptionHandler: " + errStr)
			}
			return []wasmer.Value{}, nil
		},
	)
	return function
}

func getShowSharedRWMemory(store *wasmer.Store) wasmer.IntoExtern {
	function := wasmer.NewFunction(
		store,
		wasmer.NewFunctionType(
			wasmer.NewValueTypes(),
			wasmer.NewValueTypes(),
		),
		func(args []wasmer.Value) ([]wasmer.Value, error) {
			return []wasmer.Value{}, nil
		},
	)
	return function
}

func getLog(store *wasmer.Store) wasmer.IntoExtern {
	function := wasmer.NewFunction(
		store,
		wasmer.NewFunctionType(
			wasmer.NewValueTypes(),
			wasmer.NewValueTypes(),
		),
		func(args []wasmer.Value) ([]wasmer.Value, error) {
			return []wasmer.Value{}, nil
		},
	)
	return function
}

func toArray32(s *big.Int, size int) ([]uint32, error) {
	res := make([]uint32, size)
	rem := s

	radix := big.NewInt(0x100000000)
	zero := big.NewInt(0)
	i := size - 1
	// while not zero rem
	for rem.Cmp(zero) != 0 {
		res[i] = uint32(new(big.Int).Mod(rem, radix).Uint64())
		rem.Div(rem, radix)
		i--
	}
	return res, nil
}

func fromArray32(arr []uint32) *big.Int {
	res := new(big.Int)
	radix := big.NewInt(0x100000000)
	for i := range arr {
		res.Mul(res, radix)
		res.Add(res, big.NewInt(int64(arr[i])))
	}
	return res
}

type concurrentString struct {
	sync.RWMutex
	s string
}

func (s *concurrentString) Get() string {
	s.RLock()
	v := s.s
	s.RUnlock()
	return v
}

func (s *concurrentString) Set(v string) {
	s.Lock()
	s.s = v
	s.Unlock()
}
