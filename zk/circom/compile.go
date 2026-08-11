package circom

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/pkg/errors"
)

type Compilation struct {
	R1CS        []byte
	WitnessWasm []byte
}

func Compile(body []byte, linkLibs []string) (Compilation, error) {
	dir, err := os.MkdirTemp("", "circomCompile")
	if err != nil {
		return Compilation{}, errors.Wrap(err, "")
	}
	defer os.RemoveAll(dir)

	// Write body into file.
	name := "script"
	scriptPath := filepath.Join(dir, name+".circom")
	if err := os.WriteFile(scriptPath, body, 0644); err != nil {
		return Compilation{}, errors.Wrap(err, "")
	}

	// Circom compile.
	const circomBin = "circom"
	arg := make([]string, 0, 2*len(linkLibs))
	for _, l := range linkLibs {
		arg = append(arg, []string{"-l", l}...)
	}
	arg = append(arg, []string{"--r1cs", "--sym", "--wasm"}...)
	arg = append(arg, []string{"--output", dir}...)
	arg = append(arg, scriptPath)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, circomBin, arg...)
	stdoutStderr, err := cmd.CombinedOutput()
	if err != nil {
		return Compilation{}, errors.Wrap(err, fmt.Sprintf("%s", stdoutStderr))
	}

	// Read outputs.
	cmpl := Compilation{}
	r1csPath := filepath.Join(dir, name+".r1cs")
	cmpl.R1CS, err = os.ReadFile(r1csPath)
	if err != nil {
		return Compilation{}, errors.Wrap(err, "")
	}
	wPath := filepath.Join(dir, name+"_js", name+".wasm")
	cmpl.WitnessWasm, err = os.ReadFile(wPath)
	if err != nil {
		return Compilation{}, errors.Wrap(err, "")
	}
	return cmpl, nil
}

func exportR1CS(dst, src string) error {
	const bin = "snarkjs"
	arg := []string{"r1cs", "export", "json", src, dst}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, arg...)
	stdoutStderr, err := cmd.CombinedOutput()
	if err != nil {
		return errors.Wrap(err, fmt.Sprintf("%s", stdoutStderr))
	}
	return nil
}
