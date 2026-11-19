#!/usr/bin/env bash
# Prepare the SECOND laptop (the one that joins the drone AP).
#
# That laptop loses its home Wi-Fi the moment it connects to the drone, so the
# only path back to this machine is the Ethernet cable. This script gives the
# cable link a static address and enables SSH, then prints what to do next.
#
# Run it ON THE SECOND LAPTOP:
#   sudo ./second_laptop_setup.sh
#   sudo ./second_laptop_setup.sh --key "ssh-ed25519 AAAA... user@host"
#
# `--key` appends a public key to authorized_keys for the current user so this
# machine can log in without a password.

set -uo pipefail

LINK_IP="192.168.50.2/24"   # this side of the cable
PEER_IP="192.168.50.1"      # the laptop running the agent
ETH_PROFILE="v2m-link"
PUBKEY=""

die() { printf 'Ошибка: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --key) PUBKEY="${2:-}"; shift 2 ;;
        -h|--help)
            awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"
            exit 0
            ;;
        *) die "неизвестный аргумент: $1" ;;
    esac
done

[[ $EUID -eq 0 ]] || die "запускать через sudo"

# --- 1. SSH server -----------------------------------------------------------
if ! systemctl is-active --quiet ssh 2>/dev/null \
    && ! systemctl is-active --quiet sshd 2>/dev/null; then
    printf 'Устанавливаю openssh-server...\n'
    if command -v apt-get >/dev/null; then
        apt-get update -qq && apt-get install -y -qq openssh-server
    elif command -v dnf >/dev/null; then
        dnf install -y openssh-server
    else
        die "неизвестный пакетный менеджер — установите openssh-server вручную"
    fi
fi
for unit in ssh sshd; do
    if systemctl list-unit-files 2>/dev/null | grep -q "^${unit}.service"; then
        systemctl enable --now "$unit" >/dev/null 2>&1 || true
        break
    fi
done
systemctl is-active --quiet ssh 2>/dev/null || systemctl is-active --quiet sshd 2>/dev/null \
    || die "sshd не запустился"

# --- 2. Static address on the cable -----------------------------------------
# The cable has no DHCP server, so NetworkManager must not wait for one.
# never-default keeps the drone Wi-Fi as the default route; the cable is only
# used to reach the peer /24.
printf 'Настраиваю Ethernet (%s)...\n' "$LINK_IP"
nmcli connection delete "$ETH_PROFILE" >/dev/null 2>&1 || true
ETH_DEV=$(nmcli -t -f DEVICE,TYPE device status | awk -F: '$2=="ethernet" {print $1; exit}')
[[ -n "$ETH_DEV" ]] || die "не найден Ethernet-интерфейс"

nmcli connection add type ethernet ifname "$ETH_DEV" con-name "$ETH_PROFILE" \
    ipv4.method manual \
    ipv4.addresses "$LINK_IP" \
    ipv4.never-default yes \
    ipv6.method disabled \
    connection.autoconnect yes >/dev/null || die "не удалось создать подключение"

nmcli connection up "$ETH_PROFILE" >/dev/null 2>&1 || true

# --- 3. Authorised key -------------------------------------------------------
if [[ -n "$PUBKEY" ]]; then
    TARGET_USER="${SUDO_USER:-$USER}"
    HOME_DIR=$(getent passwd "$TARGET_USER" | cut -d: -f6)
    [[ -n "$HOME_DIR" ]] || die "не найден домашний каталог для $TARGET_USER"

    install -d -m 700 -o "$TARGET_USER" -g "$TARGET_USER" "$HOME_DIR/.ssh"
    touch "$HOME_DIR/.ssh/authorized_keys"
    if grep -qF "$PUBKEY" "$HOME_DIR/.ssh/authorized_keys"; then
        printf 'Ключ уже добавлен.\n'
    else
        printf '%s\n' "$PUBKEY" >> "$HOME_DIR/.ssh/authorized_keys"
        printf 'Ключ добавлен для %s.\n' "$TARGET_USER"
    fi
    chown "$TARGET_USER:$TARGET_USER" "$HOME_DIR/.ssh/authorized_keys"
    chmod 600 "$HOME_DIR/.ssh/authorized_keys"
fi

# --- 4. Report ---------------------------------------------------------------
printf '\n'
printf 'Ethernet %s: %s\n' "$ETH_DEV" \
    "$(ip -brief addr show "$ETH_DEV" | awk '{print $3}')"
printf 'sshd: %s\n' "$(systemctl is-active ssh 2>/dev/null || systemctl is-active sshd 2>/dev/null)"
printf 'Пользователь: %s\n' "${SUDO_USER:-$USER}"
printf '\nДальше на этой машине:\n'
printf '  1) подключите дрон:  nmcli device wifi connect <SSID> password <PASS>\n'
printf '  2) проверьте:        ping -c1 10.42.0.1\n'
printf '  3) на ноуте с агентом должно пинговаться: %s\n' "$PEER_IP"
printf '\nПорт 9999/9090/22 дрона пробрасывается отдельно, менять ничего не нужно.\n'
